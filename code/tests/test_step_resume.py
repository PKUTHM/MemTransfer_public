import copy
import json
import pathlib
import sys
import tempfile
import unittest
from unittest.mock import Mock

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from step_resume import load_checkpoint, replay_checkpoint


class StepResumeTests(unittest.TestCase):
    def setUp(self):
        self.pose = dict(x=0, y=0, heading="North")
        self.after = dict(x=0, y=1, heading="North")
        self.initial = dict(current_pose=self.pose, visual_observations=dict(main_view="old.png"), stages_total=2)
        self.row = dict(record_type="decision", decision_index=1, action_seq_before=0,
            initial_origin=self.pose, pose_before=self.pose, outcome="ok", action="Move_Forward 1m",
            action_seq=1, server_status="success", server_message="moved", pose_after=self.after)
        self.response = dict(current_pose=self.after, status="success", message="moved",
            visual_observations=dict(main_view="new.png"), stages_completed=1, stages_total=2)
        self.format = lambda *args: ("history", "display")

    def test_restores_history_stage_and_marks_legacy_usage_unknown(self):
        step = Mock(return_value=self.response)
        result = replay_checkpoint([self.row], self.initial, step, self.format)
        step.assert_called_once_with("Move_Forward 1m", 1)
        self.assertEqual(result['pose'], self.after)
        self.assertEqual(result['img_history'][0]['path'], 'old.png')
        self.assertEqual(result['last_stage_completion_step'], 1)
        self.assertEqual(result['steps_with_unknown_usage'], 1)
        self.assertEqual(result['log_path'], ['history'])

    def test_replay_mismatch_fails_closed(self):
        response = dict(self.response, current_pose=self.pose)
        with self.assertRaisesRegex(ValueError, 'pose mismatch'):
            replay_checkpoint([self.row], self.initial, lambda *a: response, self.format)

    def test_technical_decision_takes_no_action_and_counters_reset_on_success(self):
        error = dict(self.row, outcome='technical', action=None)
        row = dict(self.row, decision_index=2, token_info=dict(prompt_tokens=12, completion_tokens=2, total_tokens=14, usage_reported=True))
        step = Mock(return_value=self.response)
        result = replay_checkpoint([error, row], self.initial, step, self.format)
        self.assertEqual(step.call_count, 1)
        self.assertEqual(result['technical_failures'], 1)
        self.assertEqual(result['consecutive_technical'], 0)
        self.assertEqual(result['decision_idx'], 2)
        self.assertEqual(result['token_usage']['total_tokens'], 14)
        self.assertEqual(result['steps_with_unknown_usage'], 0)

    def test_fingerprint_and_sequence_validated_before_replay(self):
        identity=dict(config_fingerprint='expected', legacy_case_id=1, group_id='R05', round_id=0, attempt_id=0)
        header=dict(identity,record_type='run_header',started_at='2026-09-13 12:00:00')
        row=dict(self.row,**{k:v for k,v in identity.items() if k!='config_fingerprint'})
        with tempfile.TemporaryDirectory() as directory:
            path=pathlib.Path(directory)/'steps.jsonl'
            path.write_text(json.dumps(header)+'\n'+json.dumps(row)+'\n',encoding='utf-8')
            self.assertEqual(len(load_checkpoint(path,identity)['rows']),1)
            with self.assertRaisesRegex(ValueError,'identity mismatch'):
                load_checkpoint(path,dict(identity,config_fingerprint='other'))
            row['decision_index']=3
            path.write_text(json.dumps(header)+'\n'+json.dumps(row)+'\n',encoding='utf-8')
            with self.assertRaisesRegex(ValueError,'non-contiguous'):
                load_checkpoint(path,identity)

    def test_terminal_cannot_have_later_decisions(self):
        terminal=dict(self.row,server_status='task_completed')
        response=dict(self.response,status='task_completed')
        with self.assertRaisesRegex(ValueError,'after terminal'):
            replay_checkpoint([terminal,self.row],self.initial,lambda *a:response,self.format)

if __name__ == '__main__':
    unittest.main()
