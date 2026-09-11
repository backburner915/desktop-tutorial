"""Session-only physical baseline. Call before initializing PhysX."""
from __future__ import annotations
import math
from pxr import Gf, Usd, UsdGeom, UsdPhysics

PD = {'arm_1_5':(400.,80.),'arm_6':(1000.,200.),'gripper':(1000.,100.)}
FRICTION_TRIALS = [{'static':.5,'dynamic':.4},{'static':.8,'dynamic':.6},{'static':1.,'dynamic':.8}]


def repair_finger_collision_schema(stage, robot):
    """Replace container-only collision opinions with stable box proxies.

    The imported R1 places ``CollisionAPI`` on an Xform container while the
    actual geometry is a nested mesh.  Enabling those high-resolution meshes
    as dynamic convex hulls can make PhysX 5.1 hang while cooking the moved
    articulation.  A box proxy uses the exact mesh world envelope (computed
    before placement) and is attached to the same rigid link, so it remains a
    finite collision shape within the visual finger envelope.
    """
    import numpy as np
    stage.SetEditTarget(Usd.EditTarget(stage.GetSessionLayer()))
    repairs=[]
    cache = UsdGeom.XformCache()
    for p in list(stage.Traverse()):
        if not p.GetPath().HasPrefix(robot.GetPath()) or 'gripper_link' not in str(p.GetPath()): continue
        if p.HasAPI(UsdPhysics.CollisionAPI) and p.IsA(UsdGeom.Xform):
            meshes=[q for q in Usd.PrimRange(p) if q.IsA(UsdGeom.Mesh)]
            if not meshes: raise ValueError('finger collider container has no mesh')
            # The collision container is authored directly below the finger
            # link (``<gripper_link>/collisions``).  Keep the replacement
            # proxy under that link so PhysX assigns contacts to the
            # independent finger rigid body; putting it at the grandparent
            # robot root silently loses per-finger contact reporting.
            link_path = p.GetPath().GetParentPath()
            link = stage.GetPrimAtPath(link_path)
            if not link or not link.IsValid():
                raise ValueError(f'finger collision link is missing: {link_path}')
            link_world = UsdGeom.XformCache().GetLocalToWorldTransform(link).RemoveScaleShear()
            link_inv = link_world.GetInverse()
            for q in meshes:
                raw_points = UsdGeom.Mesh(q).GetPointsAttr().Get()
                if raw_points is None or len(raw_points) == 0:
                    raise ValueError(f'finger collider has no vertices: {q.GetPath()}')
                mesh_world = cache.GetLocalToWorldTransform(q)
                world_points = np.asarray([mesh_world.Transform(Gf.Vec3d(*map(float, point))) for point in raw_points], dtype=np.float64)
                world_min = world_points.min(axis=0)
                world_max = world_points.max(axis=0)
                if not np.isfinite(world_min).all() or not np.isfinite(world_max).all() or np.any(world_max <= world_min):
                    raise ValueError(f'finger collider has invalid bounds: {q.GetPath()}')
                corners = np.asarray(
                    [[x, y, z] for x in (world_min[0], world_max[0]) for y in (world_min[1], world_max[1]) for z in (world_min[2], world_max[2])],
                    dtype=np.float64,
                )
                local = np.asarray([link_inv.Transform(Gf.Vec3d(*point)) for point in corners], dtype=np.float64)
                local_min = local.min(axis=0)
                local_max = local.max(axis=0)
                proxy_path = link_path.AppendChild(f'PhysicsCollisionProxy_{len(repairs)}')
                proxy = UsdGeom.Cube.Define(stage, proxy_path)
                proxy.CreateSizeAttr(1.0)
                xform = UsdGeom.Xformable(proxy.GetPrim())
                xform.ClearXformOpOrder()
                xform.AddTranslateOp(precision=UsdGeom.XformOp.PrecisionDouble).Set(Gf.Vec3d(*((local_min + local_max) / 2.0)))
                xform.AddScaleOp(precision=UsdGeom.XformOp.PrecisionFloat).Set(Gf.Vec3f(*(local_max - local_min)))
                UsdPhysics.CollisionAPI.Apply(proxy.GetPrim()).CreateCollisionEnabledAttr(True)
                repairs.append({'container':str(p.GetPath()),'mesh_source':str(q.GetPath()),'proxy':str(proxy.GetPath()),'world_bbox_m':{'min':world_min.tolist(),'max':world_max.tolist()}})
                if q.HasAPI(UsdPhysics.CollisionAPI): q.RemoveAPI(UsdPhysics.CollisionAPI)
                if q.HasAPI(UsdPhysics.MeshCollisionAPI): q.RemoveAPI(UsdPhysics.MeshCollisionAPI)
            p.RemoveAPI(UsdPhysics.CollisionAPI)
            if p.HasAPI(UsdPhysics.MeshCollisionAPI): p.RemoveAPI(UsdPhysics.MeshCollisionAPI)
    return repairs


def add_table_supports(stage, table_path, ground_z=0.):
    from r1_bimanual_dataset.tools.inspect_scene_geometry import points
    import numpy as np
    stage.SetEditTarget(Usd.EditTarget(stage.GetSessionLayer()))
    p=points(stage.GetPrimAtPath(table_path));lo=p.min(0);hi=p.max(0)
    if lo[2]<=ground_z: raise ValueError('invalid ground/table gap')
    units=UsdGeom.GetStageMetersPerUnit(stage);t=.045;h=float(lo[2]-ground_z)
    specs=[]
    for i,x in enumerate((lo[0]+t/2,hi[0]-t/2)):
        for j,y in enumerate((lo[1]+t/2,hi[1]-t/2)):
            specs.append((f'Leg{i}{j}',[x,y,ground_z+h/2],[t,t,h]))
    for j,y in enumerate((lo[1]+t/2,hi[1]-t/2)):
        specs.append((f'Crossbeam{j}',[(lo[0]+hi[0])/2,y,ground_z+h*.35],[float(hi[0]-lo[0]-2*t),t,t]))
    paths=[]
    for name,center,size in specs:
        c=UsdGeom.Cube.Define(stage,'/World/ReferenceSupports/'+name);c.CreateSizeAttr(1.)
        x=UsdGeom.Xformable(c);x.ClearXformOpOrder();x.AddTranslateOp().Set(Gf.Vec3d(*[float(v/units) for v in center]));x.AddScaleOp().Set(Gf.Vec3f(*[float(v/units) for v in size]))
        c.CreateDisplayColorAttr([Gf.Vec3f(.25,.3,.35)])
        UsdPhysics.CollisionAPI.Apply(c.GetPrim()).CreateCollisionEnabledAttr(True)
        paths.append(str(c.GetPath()))
    return paths


def fix_root(stage,robot):
    stage.SetEditTarget(Usd.EditTarget(stage.GetSessionLayer()))
    bodies={p.GetPath():p for p in Usd.PrimRange(robot) if p.HasAPI(UsdPhysics.RigidBodyAPI)}
    children=set()
    for p in Usd.PrimRange(robot):
        if p.IsA(UsdPhysics.Joint): children.update(UsdPhysics.Joint(p).GetBody1Rel().GetTargets())
    roots=set(bodies)-children
    if len(roots)!=1: raise ValueError('ambiguous root rigid body: '+str(roots))
    root=bodies[roots.pop()];pose=UsdGeom.XformCache().GetLocalToWorldTransform(root).RemoveScaleShear()
    joint=UsdPhysics.FixedJoint.Define(stage,'/World/ReferenceRootFixedJoint')
    joint.CreateBody0Rel().ClearTargets(True);joint.CreateBody1Rel().SetTargets([root.GetPath()])
    joint.CreateLocalPos0Attr(Gf.Vec3f(pose.ExtractTranslation()));joint.CreateLocalRot0Attr(Gf.Quatf(pose.ExtractRotationQuat()))
    joint.CreateLocalPos1Attr(Gf.Vec3f(0.));joint.CreateLocalRot1Attr(Gf.Quatf(1.))
    return str(root.GetPath())


def configure_physics(stage, robot, friction_index=0):
    from pxr import PhysxSchema, UsdShade
    stage.SetEditTarget(Usd.EditTarget(stage.GetSessionLayer()))
    scenes=[p for p in stage.Traverse() if p.IsA(UsdPhysics.Scene)]
    if not scenes:
        scene=UsdPhysics.Scene.Define(stage,'/World/physicsScene');scene.CreateGravityDirectionAttr(Gf.Vec3f(0,0,-1));scene.CreateGravityMagnitudeAttr(9.81/UsdGeom.GetStageMetersPerUnit(stage));scenes=[scene.GetPrim()]
    if len(scenes)>1:
        primary=stage.GetPrimAtPath('/World/PhysicsScene')
        if not primary or not primary.IsA(UsdPhysics.Scene): raise ValueError('ambiguous physics scene')
        for item in scenes:
            if item!=primary:item.SetActive(False)
        scenes=[primary]
    api=PhysxSchema.PhysxSceneAPI.Apply(scenes[0]);api.CreateTimeStepsPerSecondAttr(120);api.CreateEnableCCDAttr(True);api.CreateEnableGPUDynamicsAttr(False);api.CreateBroadphaseTypeAttr('MBP')
    friction=FRICTION_TRIALS[friction_index];mat=UsdShade.Material.Define(stage,'/World/ReferencePhysicsMaterial');m=UsdPhysics.MaterialAPI.Apply(mat.GetPrim())
    m.CreateStaticFrictionAttr(friction['static']);m.CreateDynamicFrictionAttr(friction['dynamic']);m.CreateRestitutionAttr(0.)
    for p in stage.Traverse():
        if p.HasAPI(UsdPhysics.RigidBodyAPI):PhysxSchema.PhysxRigidBodyAPI.Apply(p).CreateEnableCCDAttr(True)
        if p.HasAPI(UsdPhysics.CollisionAPI):
            c=PhysxSchema.PhysxCollisionAPI.Apply(p);c.CreateContactOffsetAttr(.002/UsdGeom.GetStageMetersPerUnit(stage));c.CreateRestOffsetAttr(0.)
            UsdShade.MaterialBindingAPI.Apply(p).Bind(mat,materialPurpose='physics')
        if p.GetPath().HasPrefix(robot.GetPath()):
            name=p.GetName()
            if 'gripper_axis' in name and p.IsA(UsdPhysics.PrismaticJoint): values=PD['gripper'];kind='linear'
            elif '_arm_joint' in name and p.IsA(UsdPhysics.RevoluteJoint): values=PD['arm_6' if name.endswith('6') else 'arm_1_5'];kind='angular'
            else: continue
            drive=UsdPhysics.DriveAPI.Apply(p,kind)
            # USD angular drives consume degrees. Convert Isaac Lab's per-radian SI gains.
            scale=math.pi/180 if kind=='angular' else 1.
            drive.CreateStiffnessAttr(values[0]*scale);drive.CreateDampingAttr(values[1]*scale)
    return {'physics_hz':120,'cpu_physx':True,'ccd':True,'gpu_dynamics':False,'contact_offset_m':.002,'rest_offset_m':0.,'restitution':0.,'friction_trial':friction,'pd_si':PD,'friction_selected':False}
