"""Static packaging checks; never execute analysis jobs or call external services."""
import argparse,ast,importlib.util,json,sys
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--code',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();roots=[a.code]+list((a.code/'third_party').glob('*/src'));modules={};issues=[];external=set();count=0
for root in roots:
 for f in root.rglob('*.py'):
  if root==a.code and any(x in f.relative_to(root).parts for x in ['third_party','tests']):continue
  mod='.'.join(f.relative_to(root).with_suffix('').parts);mod=mod.removesuffix('.__init__');modules[mod]=f
for mod,f in modules.items():
 try:t=ast.parse(f.read_text());compile(t,str(f),'exec');count+=1
 except Exception as e:issues.append(dict(module=mod,kind='syntax',detail=str(e)));continue
 for n in ast.walk(t):
  if isinstance(n,ast.ImportFrom) and n.module and not n.level:
   name=n.module
   if name.split('.')[0] in {'hrm','hrm_ired','hrm_cremp','hrm_qwen','ringer'}:
    if name not in modules and not any(x.startswith(name+'.') for x in modules):issues.append(dict(module=mod,kind='missing_internal_import',detail=name))
   elif name.split('.')[0] not in sys.stdlib_module_names:external.add(name.split('.')[0])
  if isinstance(n,ast.Constant) and isinstance(n.value,str) and n.value.startswith(('/sg-pretrain/','/Users/','/home/','/root/')):issues.append(dict(module=mod,kind='historical_absolute_path',detail=n.value))
report=dict(parsed_modules=count,issues=issues,external_import_roots=sorted(external),scope='Static syntax/import/path audit, not an end-to-end analysis validation')
a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(report,indent=2));print(json.dumps(dict(parsed=count,issues=len(issues),missing_internal=sum(x['kind']=='missing_internal_import' for x in issues))))
sys.exit(any(x['kind'] in ['syntax','missing_internal_import'] for x in issues))
