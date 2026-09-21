"""Regression: invalid game checkout must persist a sanitized failure result."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

class FailureRecordTest(unittest.TestCase):
    def test_invalid_checkout_is_recorded_without_private_path(self):
        with tempfile.TemporaryDirectory(prefix='private-audit-input-') as folder:
            output=Path(folder)/'failure.json'
            process=subprocess.run([sys.executable,str(Path(__file__).with_name('audit_controls.py')),
                '--game',str(Path(folder)/'missing-checkout'),
                '--output',str(output)],capture_output=True,text=True)
            self.assertEqual(process.returncode,1)
            text=output.read_text(encoding='utf8')
            result=json.loads(text)
            self.assertEqual(result['status'],'error')
            self.assertEqual(result['error']['phase'],'validate_game_checkout')
            self.assertNotIn('private-audit-input-',text)
            self.assertNotIn('missing-checkout',text)
            self.assertGreaterEqual(result['timings']['total_wall_seconds'],0)
            self.assertFalse(result['peak_vram']['applicable'])
            self.assertFalse(result['probabilities']['applicable'])
            self.assertTrue(result['environment']['python'])
            self.assertEqual(result['cases'],[])

if __name__=='__main__': unittest.main()
