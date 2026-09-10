"""USD geometry audit; no IK, simulation, or dataset writes."""
from __future__ import annotations
import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[2]
PKG = ROOT / 'r1_bimanual_dataset'
if str(ROOT) not in sys.path: sys.path.insert(0, str(ROOT))


def bootstrap_usd():
    try:
        from pxr import Usd
        return
    except ImportError:
        pass
    base = Path(sys.prefix) / 'Lib/site-packages/isaacsim'
    libs = next((base / 'extscache').glob('omni.usd.libs-*'))
    sys.path.insert(0, str(libs))
    os.environ['PXR_USD_WINDOWS_DLL_PATH'] = os.pathsep.join(str(p) for p in (base/'kit', base/'kit/kernel/plugins', libs/'bin') if p.exists())


def points(prim):
    from pxr import Gf, UsdGeom
    if prim.IsA(UsdGeom.Mesh):
        raw = UsdGeom.Mesh(prim).GetPointsAttr().Get()
        if raw is None or len(raw) == 0: raise ValueError('mesh has no points: '+str(prim.GetPath()))
    elif prim.IsA(UsdGeom.Cube):
        h = UsdGeom.Cube(prim).GetSizeAttr().Get()/2
        raw = [(x*h,y*h,z*h) for x in (-1,1) for y in (-1,1) for z in (-1,1)]
    else:
        raise ValueError('unsupported collider geometry: '+str(prim.GetPath()))
    matrix = UsdGeom.XformCache().GetLocalToWorldTransform(prim)
    result = np.array([matrix.Transform(Gf.Vec3d(*map(float,p))) for p in raw]) * UsdGeom.GetStageMetersPerUnit(prim.GetStage())
    if not np.isfinite(result).all(): raise ValueError('nonfinite geometry: '+str(prim.GetPath()))
    return result


def bbox(p):
    return {'min_m':p.min(0).tolist(),'max_m':p.max(0).tolist(),'dimensions_m':np.ptp(p,axis=0).tolist()}


def _inner_surface_patch(collision_prims, link_matrix, axis_local):
    """Extract the largest planar surface facing the opposing finger.

    A gripper-link collision mesh also contains its mount and back side.  Its
    full projection is therefore not an aperture measurement.  The inward
    facing surface is selected from mesh face normals in the link frame and
    clustered by its coordinate along the prismatic axis.  This keeps the
    measurement tied to authored collision geometry while excluding the
    unrelated back and tip surfaces.
    """
    from pxr import Gf, UsdGeom

    axis_local = np.asarray(axis_local, dtype=np.float64)
    axis_local /= max(np.linalg.norm(axis_local), 1e-12)
    inward = -axis_local
    cache = UsdGeom.XformCache()
    link_inverse = link_matrix.GetInverse()
    rows = []
    for prim in collision_prims:
        if not prim.IsA(UsdGeom.Mesh):
            continue
        mesh = UsdGeom.Mesh(prim)
        raw_points = np.asarray(mesh.GetPointsAttr().Get(), dtype=np.float64)
        indices = np.asarray(mesh.GetFaceVertexIndicesAttr().Get(), dtype=np.int64)
        counts = np.asarray(mesh.GetFaceVertexCountsAttr().Get(), dtype=np.int64)
        if raw_points.size == 0 or indices.size == 0 or counts.size == 0:
            continue
        mesh_to_link = link_inverse * cache.GetLocalToWorldTransform(prim)
        local_points = np.asarray(
            [mesh_to_link.Transform(Gf.Vec3d(*p)) for p in raw_points], dtype=np.float64
        )
        normals = mesh.GetNormalsAttr().Get()
        raw_normals = np.asarray(normals, dtype=np.float64) if normals is not None else None
        offset = 0
        for count in counts:
            ids = indices[offset : offset + int(count)]
            verts = local_points[ids]
            if len(verts) < 3:
                offset += int(count)
                continue
            if raw_normals is not None and len(raw_normals) >= offset + int(count):
                normal = raw_normals[offset : offset + int(count)].mean(axis=0)
                normal = np.asarray(mesh_to_link.TransformDir(Gf.Vec3d(*normal)), dtype=np.float64)
            else:
                normal = np.cross(verts[1] - verts[0], verts[2] - verts[0])
            norm = float(np.linalg.norm(normal))
            if norm <= 1e-9:
                offset += int(count)
                continue
            normal /= norm
            area = 0.0
            for i in range(1, len(verts) - 1):
                area += float(np.linalg.norm(np.cross(verts[i] - verts[0], verts[i + 1] - verts[0])) / 2.0)
            if float(np.dot(normal, inward)) > 0.90 and area > 1e-12:
                rows.append((float(np.dot(verts.mean(axis=0), axis_local)), area, verts, normal))
            offset += int(count)
    if not rows:
        raise ValueError("no inward-facing collision surface found")
    bins: dict[float, float] = {}
    for projection, area, _, _ in rows:
        key = round(projection / 0.0005) * 0.0005
        bins[key] = bins.get(key, 0.0) + area
    plane = max(bins, key=bins.get)
    selected = [row for row in rows if abs(row[0] - plane) <= 0.0015]
    if not selected:
        raise ValueError("inward collision surface cluster is empty")
    patch_local = np.concatenate([row[2] for row in selected], axis=0)
    patch_world = np.asarray(
        [link_matrix.Transform(Gf.Vec3d(*p)) for p in patch_local], dtype=np.float64
    )
    normal_world = np.asarray(link_matrix.TransformDir(Gf.Vec3d(*selected[0][3])), dtype=np.float64)
    normal_world /= max(np.linalg.norm(normal_world), 1e-12)
    return patch_world, normal_world, {
        "local_plane_coordinate_m": plane,
        "face_count": len(selected),
        "area_m2": float(sum(row[1] for row in selected)),
        "normal_local": selected[0][3].tolist(),
    }


def audit(stage, root):
    from pxr import Usd, UsdGeom, UsdPhysics, UsdShade
    visual=[]; collisions=[]; records=[]; errors=[]
    for prim in Usd.PrimRange.Stage(stage, Usd.TraverseInstanceProxies()):
        if not prim.GetPath().HasPrefix(root.GetPath()): continue
        collision=prim.HasAPI(UsdPhysics.CollisionAPI)
        record={'path':str(prim.GetPath()),'parent':str(prim.GetParent().GetPath()),'type':prim.GetTypeName(), 'RigidBodyAPI':prim.HasAPI(UsdPhysics.RigidBodyAPI),'CollisionAPI':collision,'MassAPI':prim.HasAPI(UsdPhysics.MassAPI)}
        for key,attr in {'mass_kg':'physics:mass','density_kg_m3':'physics:density','contact_offset_m':'physxCollision:contactOffset','rest_offset_m':'physxCollision:restOffset','ccd_enabled':'physxRigidBody:enableCCD','collision_enabled':'physics:collisionEnabled'}.items():
            a=prim.GetAttribute(attr);record[key]=a.Get() if a else None
        record['material']={}
        material,_=UsdShade.MaterialBindingAPI(prim).ComputeBoundMaterial('physics')
        if material:
            record['material']['path']=str(material.GetPath())
            for key in ['staticFriction','dynamicFriction','restitution','density']:
                a=material.GetPrim().GetAttribute('physics:'+key)
                record['material'][key]=a.Get() if a else None
        if collision and record['collision_enabled'] is not False:
            try:
                p=points(prim);record['bbox']=bbox(p);collisions.append((str(prim.GetPath()),p))
            except ValueError as exc: errors.append(str(exc))
        if prim.IsA(UsdGeom.Gprim) and UsdGeom.Imageable(prim).ComputeVisibility() != 'invisible' and UsdGeom.Imageable(prim).ComputePurpose() in ('default','render'):
            try: visual.append(points(prim))
            except ValueError as exc: errors.append(str(exc))
        if collision or record['RigidBodyAPI'] or record['MassAPI']: records.append(record)
    if not visual or not collisions: raise ValueError('missing visual geometry or enabled colliders')
    v=np.concatenate(visual);c=np.concatenate([p for _,p in collisions])
    lo,hi=v.min(0),v.max(0)
    violations=[name for name,p in collisions if np.any(p.min(0)<lo-.03) or np.any(p.max(0)>hi+.03)]
    error=float(max(np.max(np.abs(c.min(0)-lo)),np.max(np.abs(c.max(0)-hi))))
    return {'visual_bbox':bbox(v),'collision_bbox':bbox(c),'collider_count':len(collisions),'hierarchy':records,'envelope_error_m':error,'colliders_outside_visual_envelope':violations,'geometry_errors':errors,'geometry_pass':not errors and not violations and error<=.03,'property_note':'null means unauthored/unresolved; never interpreted as zero. Bounds use mesh vertices, not authored extent hints.'}


def grippers(stage,robot):
    from pxr import Gf, Usd, UsdGeom, UsdPhysics
    result={}
    for side in ['left','right']:
        pads=[];origins=[];joint_info=[];endpoint_clouds={'lower_m':[],'upper_m':[]};endpoint_origins={'lower_m':[],'upper_m':[]}
        for number in (1,2):
            links=[p for p in stage.Traverse() if p.GetPath().HasPrefix(robot.GetPath()) and p.GetName()==f'{side}_gripper_link{number}' and p.HasAPI(UsdPhysics.RigidBodyAPI)]
            if len(links)!=1: raise ValueError('ambiguous/missing finger link')
            link=links[0]
            cloud=[points(p) for p in Usd.PrimRange(link,Usd.TraverseInstanceProxies()) if p.HasAPI(UsdPhysics.CollisionAPI) and UsdPhysics.CollisionAPI(p).GetCollisionEnabledAttr().Get() is not False]
            if not cloud: raise ValueError('no finger collider')
            joints=[UsdPhysics.PrismaticJoint(p) for p in stage.Traverse() if p.IsA(UsdPhysics.PrismaticJoint) and link.GetPath() in UsdPhysics.PrismaticJoint(p).GetBody1Rel().GetTargets()]
            if len(joints)!=1: raise ValueError('finger must have one prismatic parent joint')
            j=joints[0];body0=stage.GetPrimAtPath(j.GetBody0Rel().GetTargets()[0]);cache=UsdGeom.XformCache()
            m0=cache.GetLocalToWorldTransform(body0);m1=cache.GetLocalToWorldTransform(link)
            axis=np.eye(3)['XYZ'.index(j.GetAxisAttr().Get())]
            axis=np.array(m0.TransformDir(Gf.Rotation(j.GetLocalRot0Attr().Get()).TransformDir(Gf.Vec3d(*axis))))
            axis/=np.linalg.norm(axis)
            anchor0=np.array(m0.Transform(Gf.Vec3d(j.GetLocalPos0Attr().Get())))
            anchor1=np.array(m1.Transform(Gf.Vec3d(j.GetLocalPos1Attr().Get())))
            units=UsdGeom.GetStageMetersPerUnit(stage)
            current=float(np.dot(anchor1-anchor0,axis))*units
            upper=float(j.GetUpperLimitAttr().Get())*units
            if not np.isfinite(upper) or upper>.5: raise ValueError('unbounded finger travel')
            local_cloud=np.array([m1.GetInverse().Transform(Gf.Vec3d(*map(float,p/units))) for p in np.concatenate(cloud)])
            frame0=Gf.Matrix4d(1.);frame0.SetRotate(Gf.Quatd(j.GetLocalRot0Attr().Get()));frame0.SetTranslateOnly(Gf.Vec3d(j.GetLocalPos0Attr().Get()))
            frame1=Gf.Matrix4d(1.);frame1.SetRotate(Gf.Quatd(j.GetLocalRot1Attr().Get()));frame1.SetTranslateOnly(Gf.Vec3d(j.GetLocalPos1Attr().Get()))
            for endpoint,value in [('lower_m',float(j.GetLowerLimitAttr().Get())),('upper_m',float(j.GetUpperLimitAttr().Get()))]:
                travel=Gf.Matrix4d(1.);travel.SetTranslate(Gf.Vec3d(*map(float,np.eye(3)['XYZ'.index(j.GetAxisAttr().Get())]*value)))
                desired=frame1.GetInverse()*travel*frame0*m0
                endpoint_clouds[endpoint].append(np.array([desired.Transform(Gf.Vec3d(*map(float,p))) for p in local_cloud])*units)
                endpoint_origins[endpoint].append(np.array(desired.ExtractTranslation())*units)
            pads.append(endpoint_clouds['upper_m'][-1]);origins.append(endpoint_origins['upper_m'][-1])
            joint_info.append({'path':str(j.GetPath()),'lower_m':float(j.GetLowerLimitAttr().Get())*units,'upper_m':upper,'current_geometric_m':current,'axis_world':axis.tolist()})
        # Measure both legal endpoints; do not assume positive joint travel opens.
        reference_axis=np.asarray(joint_info[0]['axis_world'])
        endpoint_options=[]
        for endpoint in ('lower_m','upper_m'):
            moved=endpoint_clouds[endpoint]
            a=reference_axis.copy()
            if np.dot(a,moved[1].mean(0)-moved[0].mean(0))<0:a=-a
            gap=float((moved[1]@a).min()-(moved[0]@a).max())
            endpoint_options.append((gap,endpoint,moved))
        _,open_endpoint,pads=max(endpoint_options,key=lambda x:x[0])
        origins=endpoint_origins[open_endpoint]
        axis=np.asarray(joint_info[0]['axis_world'])
        if np.dot(axis,pads[1].mean(0)-pads[0].mean(0))<0: axis=-axis
        inner0=float((pads[0]@axis).max());inner1=float((pads[1]@axis).min());width=inner1-inner0
        allp=np.concatenate(pads);center=allp.mean(0);center+=axis*((inner0+inner1)/2-center@axis)
        residual=allp-center;residual-=np.outer(residual@axis,axis)
        _,_,vh=np.linalg.svd(residual,full_matrices=False);approach=vh[0]
        if np.dot(approach,center-np.mean(origins,axis=0))<0: approach=-approach
        transverse=np.cross(axis,approach);transverse/=np.linalg.norm(transverse)
        approach=np.cross(transverse,axis)
        whole_finger_projection_gap=width
        tip_regions=[]
        for cloud in pads:
            projected=cloud@approach
            depth=min(.02,float(np.ptp(projected))*.25)
            region=cloud[projected>=projected.max()-depth]
            if len(region)<3: raise ValueError('insufficient fingertip collision surface vertices')
            tip_regions.append(region)
        inner0=float((tip_regions[0]@axis).max());inner1=float((tip_regions[1]@axis).min());width=inner1-inner0
        center=np.concatenate(tip_regions).mean(0);center+=axis*((inner0+inner1)/2-center@axis)
        tip=center+approach*float(np.max((np.concatenate(tip_regions)-center)@approach))
        result[side]={'closing_axis':axis.tolist(),'inner_planes':[{'normal':axis.tolist(),'offset_m':inner0},{'normal':axis.tolist(),'offset_m':inner1}],'fingertip_center_m':tip.tolist(),'contact_center_m':center.tolist(),'approach_normal':approach.tolist(),'effective_opening_m':width if width>0 else None,'signed_pad_plane_gap_m':width,'whole_finger_projection_gap_m':whole_finger_projection_gap,'pad_extraction':'distal 25 percent capped at 20 mm; collision vertices; needs surface/contact validation','station_footprint_m':float(np.ptp(allp@transverse)),'finger_length_m':float(np.ptp(allp@approach)),'open_endpoint':open_endpoint,'endpoint_apertures_m':{e:g for g,e,_ in endpoint_options},'joint_geometry':joint_info,'method':'joint-frame-consistent collision vertex support planes at both prismatic limits; conservative whole-finger clearance; not link6 TCP','contact_frame_status':'NOMINAL_REQUIRES_DYNAMIC_CONTACT_VALIDATION'}
    return result


def grippers_v2(stage, robot):
    """Measure the inward pad planes from the actual collision meshes.

    The imported finger mesh includes the mount and back of the finger.  The
    legacy ``grippers`` routine projects that entire mesh and can report a
    negative aperture even when the opposing pad planes are separated.  This
    routine uses the largest authored face cluster whose normal points inward
    from each finger in its joint-local frame, then moves that patch through
    both legal prismatic endpoints.
    """
    from pxr import Gf, Usd, UsdGeom, UsdPhysics
    cache = UsdGeom.XformCache()
    result = {}
    for side in ("left", "right"):
        endpoint_patches = {"lower_m": [], "upper_m": []}
        endpoint_origins = {"lower_m": [], "upper_m": []}
        joint_records = []
        patch_records = []
        for number in (1, 2):
            link_path = f"{robot.GetPath()}/{side}_gripper_link{number}"
            link = stage.GetPrimAtPath(link_path)
            if not link or not link.IsValid() or not link.HasAPI(UsdPhysics.RigidBodyAPI):
                raise ValueError(f"missing rigid finger link: {link_path}")
            collision_prims = [
                p for p in Usd.PrimRange(link, Usd.TraverseInstanceProxies())
                if p.IsA(UsdGeom.Mesh)
                and (
                    p.HasAPI(UsdPhysics.CollisionAPI)
                    or "/collisions/" in str(p.GetPath())
                )
                and p.GetAttribute("physics:collisionEnabled").Get() is not False
            ]
            if not collision_prims:
                raise ValueError(f"no mesh collision under {link_path}")
            joints = [
                UsdPhysics.PrismaticJoint(p) for p in stage.Traverse()
                if p.IsA(UsdPhysics.PrismaticJoint)
                and link.GetPath() in UsdPhysics.PrismaticJoint(p).GetBody1Rel().GetTargets()
            ]
            if len(joints) != 1:
                raise ValueError(f"expected one prismatic joint for {link_path}")
            joint = joints[0]
            body0 = stage.GetPrimAtPath(joint.GetBody0Rel().GetTargets()[0])
            m0 = cache.GetLocalToWorldTransform(body0)
            m1 = cache.GetLocalToWorldTransform(link)
            axis_index = "XYZ".index(joint.GetAxisAttr().Get())
            axis0 = np.eye(3)[axis_index]
            axis_world = np.asarray(m0.TransformDir(Gf.Rotation(joint.GetLocalRot0Attr().Get()).TransformDir(Gf.Vec3d(*axis0))), dtype=np.float64)
            axis_world /= max(np.linalg.norm(axis_world), 1e-12)
            axis_local = np.asarray(m1.GetInverse().TransformDir(Gf.Vec3d(*axis_world)), dtype=np.float64)
            axis_local /= max(np.linalg.norm(axis_local), 1e-12)
            inward_local = -axis_local
            rows = []
            for prim in collision_prims:
                mesh = UsdGeom.Mesh(prim)
                raw_points = np.asarray(mesh.GetPointsAttr().Get(), dtype=np.float64)
                indices = np.asarray(mesh.GetFaceVertexIndicesAttr().Get(), dtype=np.int64)
                counts = np.asarray(mesh.GetFaceVertexCountsAttr().Get(), dtype=np.int64)
                normals = mesh.GetNormalsAttr().Get()
                raw_normals = np.asarray(normals, dtype=np.float64) if normals is not None else None
                mesh_to_link = m1.GetInverse() * cache.GetLocalToWorldTransform(prim)
                local_points = np.asarray([mesh_to_link.Transform(Gf.Vec3d(*p)) for p in raw_points], dtype=np.float64)
                cursor = 0
                for count in counts:
                    count = int(count)
                    ids = indices[cursor : cursor + count]
                    verts = local_points[ids]
                    cursor += count
                    if len(verts) < 3:
                        continue
                    normal = raw_normals[cursor - count : cursor].mean(axis=0) if raw_normals is not None and len(raw_normals) >= cursor else np.cross(verts[1] - verts[0], verts[2] - verts[0])
                    normal = np.asarray(mesh_to_link.TransformDir(Gf.Vec3d(*normal)), dtype=np.float64)
                    norm = float(np.linalg.norm(normal))
                    if norm <= 1e-9 or float(np.dot(normal / norm, inward_local)) <= 0.90:
                        continue
                    normal /= norm
                    area = sum(float(np.linalg.norm(np.cross(verts[i] - verts[0], verts[i + 1] - verts[0])) / 2.0) for i in range(1, len(verts) - 1))
                    if area > 1e-12:
                        rows.append((float(np.dot(verts.mean(axis=0), axis_local)), area, verts))
            if not rows:
                raise ValueError(f"no inward-facing pad surface under {link_path}")
            area_by_bin = {}
            for projection, area, _ in rows:
                key = round(projection / 0.0005) * 0.0005
                area_by_bin[key] = area_by_bin.get(key, 0.0) + area
            plane = max(area_by_bin, key=area_by_bin.get)
            selected = [row for row in rows if abs(row[0] - plane) <= 0.0015]
            patch_local = np.concatenate([row[2] for row in selected], axis=0)
            patch_world = np.asarray([m1.Transform(Gf.Vec3d(*p)) for p in patch_local], dtype=np.float64)
            anchor0 = np.asarray(m0.Transform(Gf.Vec3d(*joint.GetLocalPos0Attr().Get())), dtype=np.float64)
            anchor1 = np.asarray(m1.Transform(Gf.Vec3d(*joint.GetLocalPos1Attr().Get())), dtype=np.float64)
            current = float(np.dot(anchor1 - anchor0, axis_world))
            lower = float(joint.GetLowerLimitAttr().Get())
            upper = float(joint.GetUpperLimitAttr().Get())
            if not np.isfinite(lower + upper) or upper - lower <= 0.0 or upper - lower > 0.5:
                raise ValueError(f"invalid prismatic limits for {joint.GetPath()}")
            for endpoint, value in (("lower_m", lower), ("upper_m", upper)):
                delta = (value - current) * axis_world
                endpoint_patches[endpoint].append(patch_world + delta)
                endpoint_origins[endpoint].append(anchor1 + delta)
            patch_records.append({"link": str(link.GetPath()), "face_count": len(selected), "area_m2": float(sum(row[1] for row in selected)), "local_plane_coordinate_m": plane, "axis_local": axis_local.tolist(), "inward_normal_local": inward_local.tolist()})
            joint_records.append({"path": str(joint.GetPath()), "lower_m": lower, "upper_m": upper, "current_geometric_m": current, "axis_world": axis_world.tolist()})
        options = []
        for endpoint in ("lower_m", "upper_m"):
            p0, p1 = endpoint_patches[endpoint]
            axis = np.asarray(joint_records[0]["axis_world"], dtype=np.float64)
            if np.dot(axis, p1.mean(axis=0) - p0.mean(axis=0)) < 0.0:
                axis = -axis
            gap = float(np.min(p1 @ axis) - np.max(p0 @ axis))
            options.append((gap, endpoint, axis))
        gap, open_endpoint, axis = max(options, key=lambda item: item[0])
        patches = endpoint_patches[open_endpoint]
        origins = endpoint_origins[open_endpoint]
        inner0 = float(np.max(patches[0] @ axis)); inner1 = float(np.min(patches[1] @ axis)); opening = inner1 - inner0
        contact_center = np.concatenate(patches).mean(axis=0); contact_center += axis * ((inner0 + inner1) / 2.0 - float(np.dot(contact_center, axis)))
        residual = np.concatenate(patches) - contact_center; residual -= np.outer(residual @ axis, axis)
        _, _, vh = np.linalg.svd(residual, full_matrices=False)
        approach = vh[0]
        if np.dot(approach, contact_center - np.mean(origins, axis=0)) < 0.0: approach = -approach
        transverse = np.cross(axis, approach); transverse /= max(np.linalg.norm(transverse), 1e-12); approach = np.cross(transverse, axis)
        result[side] = {
            "closing_axis": axis.tolist(), "inner_planes": [{"normal": axis.tolist(), "offset_m": inner0}, {"normal": axis.tolist(), "offset_m": inner1}],
            "fingertip_center_m": contact_center.tolist(), "contact_center_m": contact_center.tolist(), "approach_normal": approach.tolist(),
            "effective_opening_m": opening if opening > 0 else None, "signed_pad_plane_gap_m": opening, "whole_finger_projection_gap_m": float(gap),
            "pad_extraction": "largest inward-facing collision face cluster by triangle area; joint-local -axis normal",
            "pad_surface": patch_records, "station_footprint_m": float(np.ptp(np.concatenate(patches) @ transverse)),
            "open_endpoint": open_endpoint, "endpoint_apertures_m": {name: value for value, name, _ in options}, "joint_geometry": joint_records,
            "method": "joint-frame-consistent inward collision surface planes at both prismatic limits", "contact_frame_status": "NOMINAL_REQUIRES_DYNAMIC_CONTACT_VALIDATION",
        }
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--candidate',required=True,help='sugar_box, cracker_box, or USD path')
    parser.add_argument('--scene-config',type=Path,default=ROOT/'scene_config.json')
    parser.add_argument('--output',type=Path)
    args=parser.parse_args();name=args.candidate
    asset={'sugar_box':PKG/'assets/004_sugar_box_physics.usd','cracker_box':PKG/'assets/003_cracker_box_physics.usd'}.get(name,Path(name))
    out=args.output or ROOT/'reports'/('candidate_'+asset.stem+'.json')
    report={'candidate':name,'asset':str(asset),'status':'REJECTED','gate':'A','dataset_allowed':False,'static_settle':{'status':'NOT_RUN'},'collision_visualization':{'status':'NOT_RUN'},'source_usd_modified':False}
    try:
        from r1_bimanual_dataset.core.reference_policy import require_candidate
        require_candidate(asset)
        if not asset.is_file(): raise FileNotFoundError(str(asset))
        bootstrap_usd()
        from pxr import Usd
        cfg=json.loads(args.scene_config.read_text(encoding='utf-8'))
        stage=Usd.Stage.Open(str(asset));report.update(audit(stage,stage.GetPseudoRoot()))
        scene=Usd.Stage.Open(cfg['stage_path'])
        from r1_bimanual_dataset.core.qualified_scene import repair_finger_collision_schema
        report['session_finger_schema_repairs']=repair_finger_collision_schema(scene,scene.GetPrimAtPath(cfg['robot_prim']))
        report['grippers']=grippers_v2(scene,scene.GetPrimAtPath(cfg['robot_prim']))
        dims=np.array(report['collision_bbox']['dimensions_m']);thickness=float(dims.min());long_axis=int(dims.argmax());length=float(dims.max())
        clearance=.005;g=report['grippers'];minimum_spacing=(g['left']['station_footprint_m']+g['right']['station_footprint_m'])/2+clearance
        available_spacing=length-(g['left']['station_footprint_m']+g['right']['station_footprint_m'])/2-clearance
        report.update(object_grasp_thickness_m=thickness,safety_margin_m=clearance,minimum_station_spacing_m=minimum_spacing,available_station_spacing_m=available_spacing,long_axis=long_axis,gripper_fit_pass=all(x['effective_opening_m'] is not None and x['effective_opening_m']>thickness+clearance for x in g.values()),station_fit_pass=available_spacing>minimum_spacing)
        report['gripper_fit_method']='Distal collision vertex support planes in joint-consistent orientation. Extracted pad region requires surface/contact validation; whole-finger overlap reported separately.'
        report['grasp_stations_status']='PROVISIONAL_FINGER_FOOTPRINT_ONLY; complete hand collision not validated'
        report['rejection_reasons']=[]
        if not report['geometry_pass']:report['rejection_reasons'].append('COLLISION_VISUAL_ENVELOPE')
        if not report['gripper_fit_pass']:report['rejection_reasons'].append('EFFECTIVE_APERTURE_NOT_CERTIFIED')
        if not report['station_fit_pass']:report['rejection_reasons'].append('INSUFFICIENT_STATION_CLEARANCE')
        report['asset_sha256']=hashlib.sha256(asset.read_bytes()).hexdigest()
        report['status']='PENDING_STATIC_SETTLE_AND_VISUALIZATION' if report['geometry_pass'] and report['gripper_fit_pass'] and report['station_fit_pass'] else 'REJECTED'
    except Exception as exc:
        report['error']=type(exc).__name__+': '+str(exc)
    out.parent.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps(report,indent=2,ensure_ascii=False,allow_nan=False),encoding='utf-8')
    print(json.dumps({'status':report['status'],'report':str(out),'error':report.get('error'),'geometry_pass':report.get('geometry_pass'),'gripper_fit_pass':report.get('gripper_fit_pass'),'station_fit_pass':report.get('station_fit_pass')},ensure_ascii=False))
    return 2

if __name__=='__main__': raise SystemExit(main())
