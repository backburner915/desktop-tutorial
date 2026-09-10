"""Gate A diagnostic: fixed scene, 1200 physics steps, no episodes."""
import sys,json,hashlib,traceback
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from r1_bimanual_dataset.core.launch import prepare_isaac_environment,simulation_app_config

def main():
    cfg=json.loads((ROOT/'scene_config.json').read_text(encoding='utf-8'))
    out=ROOT/'reports/fixed_scene_static_audit.json'
    report={'status':'INVALID','duration_required_s':10,'dataset_episodes_written':0,'source_usd_modified':False}
    source=Path(cfg['stage_path']);before=hashlib.sha256(source.read_bytes()).hexdigest();app=None
    try:
        prepare_isaac_environment()
        from isaacsim import SimulationApp
        app=SimulationApp(simulation_app_config(True,'None'))
        from pxr import Usd,UsdGeom
        from r1_bimanual_dataset.tools.physical_reference_probe import _open_stage
        from r1_bimanual_dataset.core.qualified_scene import repair_finger_collision_schema,add_table_supports,fix_root,configure_physics
        stage,timeline=_open_stage(app,cfg['stage_path']);robot=stage.GetPrimAtPath(cfg['robot_prim'])
        report['finger_schema_repairs']=repair_finger_collision_schema(stage,robot)
        report['supports']=add_table_supports(stage,'/World/TaskSetup/Fixtures/StorageRack/Top')
        report['fixed_root']=fix_root(stage,robot)
        report['baseline']=configure_physics(stage,robot)
        # Object fit is a prior hard gate: no candidate is admitted for this fixture-only diagnostic.
        report['target']='source T01, fixture diagnostic only; not a Reference candidate'
        from isaacsim.core.api import SimulationContext
        sim=SimulationContext(physics_dt=1/120,rendering_dt=1/120,backend='numpy',device='cpu',physics_prim_path='/World/PhysicsScene')
        sim.initialize_physics();sim.play()
        from isaacsim.core.prims import SingleArticulation, SingleRigidPrim
        articulation=SingleArticulation(cfg['robot_prim']);articulation.initialize()
        names=articulation.dof_names
        from r1_bimanual_dataset.config import ACTION_JOINT_NAMES
        indices=np.array([names.index(n) for n in ACTION_JOINT_NAMES])
        targets=np.array([cfg['home_joint_positions'][n] for n in ACTION_JOINT_NAMES])
        from isaacsim.core.utils.types import ArticulationAction
        target=SingleRigidPrim(cfg['target_object']);target.initialize()
        traces=[]
        initial_root=articulation.get_world_pose();initial_object=target.get_world_pose()
        table=stage.GetPrimAtPath('/World/TaskSetup/Fixtures/StorageRack/Top')
        initial_table=np.array(UsdGeom.XformCache().GetLocalToWorldTransform(table))
        for step in range(1200):
            articulation.apply_action(ArticulationAction(joint_positions=targets,joint_indices=indices))
            sim.step(render=False)
            rp,rq=articulation.get_world_pose();op,oq=target.get_world_pose()
            traces.append([*map(float,rp),*map(float,rq),*map(float,op),*map(float,oq)])
        values=np.array(traces)
        report['duration_s']=1200/120;report['physics_steps']=1200
        report['root_max_translation_m']=float(np.max(np.linalg.norm(values[:,:3]-np.asarray(initial_root[0]),axis=1)))
        report['source_target_max_translation_m']=float(np.max(np.linalg.norm(values[:,7:10]-np.asarray(initial_object[0]),axis=1)))
        report['finite']=bool(np.isfinite(values).all());report['table_transform_unchanged']=bool(np.array_equal(initial_table,np.array(UsdGeom.XformCache().GetLocalToWorldTransform(table))))
        report['status']='PENDING_PENETRATION_AND_CONTACT_AUDIT' if report['finite'] and report['root_max_translation_m']<.001 and report['source_target_max_translation_m']<.01 else 'FAIL'
        report['penetration_verified']=False
        (ROOT/'reports/fixed_scene_static_trace.json').write_text(json.dumps(traces),encoding='utf-8')
    except Exception as exc:
        report['error']=type(exc).__name__+': '+str(exc);report['traceback']=traceback.format_exc()
    finally:
        report['source_usd_sha256_before']=before;report['source_usd_sha256_after']=hashlib.sha256(source.read_bytes()).hexdigest()
        report['source_usd_modified']=report['source_usd_sha256_after']!=before
        out.write_text(json.dumps(report,indent=2,ensure_ascii=False,allow_nan=False),encoding='utf-8')
        print('STATIC_AUDIT='+json.dumps(report,ensure_ascii=False),flush=True)
        if app:app.close()
    return 2
if __name__=='__main__':raise SystemExit(main())
