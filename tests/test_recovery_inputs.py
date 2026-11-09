import json
from pathlib import Path
import tempfile
import unittest
from hrm.analyse.dynamics.perturbation_recovery.inputs import arguments


class RecoveryInputTests(unittest.TestCase):
    def fixture(self, root):
        trajectory=root/'trajectory';trajectory.mkdir()
        (root/'config.json').write_text(json.dumps(dict(seed=1794199999,steps=8,anchors=[4],radii=[.01],measure_steps=[0,8],coordinate_steps=[0,4,8])))
        (trajectory/'summary.json').write_text(json.dumps(dict(selected=[dict(case=17),dict(case=33)],model=dict(checkpoint_sha256='a'*64))))
        for name in ['initial.npz']+[f'H{h:03d}_{suffix}' for h in range(1,6) for suffix in ('L_bits.npy','context.npz')]:
            (trajectory/name).touch()
        return ['--records',str(root),'--case','33']

    def test_paths_case_and_optional_reference(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);argv=self.fixture(root);a,cfg,summary,index=arguments(argv)
            self.assertEqual(index,1);self.assertEqual(a.output_dir,root/'recovery');self.assertIsNone(a.reference)
            self.assertFalse(a.output_dir.exists())
            with self.assertRaises(FileNotFoundError):arguments(argv+['--reference',str(root/'missing.npz')])
            with self.assertRaises(FileNotFoundError):arguments(argv,landscape=True)
            (root/'global_projection.npz').touch();self.assertEqual(arguments(argv,landscape=True)[0].output_dir,root/'landscape_recovery')

    def test_missing_explicit_input_never_falls_back(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);argv=self.fixture(root)
            for option in ['--trajectory','--run-config','--checkpoint']:
                with self.assertRaises(FileNotFoundError):arguments(argv+[option,str(root/'missing')])
            (root/'recovery').mkdir()
            with self.assertRaises(FileExistsError):arguments(argv)

    def test_bad_case_and_schedule(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);argv=self.fixture(root)
            with self.assertRaises(ValueError):arguments(argv+['--case','999'])
            path=root/'config.json';cfg=json.loads(path.read_text())
            for key,value in [('seed',None),('anchors',[0]),('radii',[0]),('measure_steps',[1,8]),('coordinate_steps',[0,9])]:
                path.write_text(json.dumps(dict(cfg,**{key:value})))
                with self.assertRaises(ValueError):arguments(argv)
