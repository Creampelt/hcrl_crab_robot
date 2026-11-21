"""Build ``turksat_orbit.urdf``: the Turksat model as ONE rigid link for orbit flight.

The CAD export puts all 2411.71 kg in one link 4 m from a massless root, with 44 links of 1e-9 kg on fixed
joints. This collapses every visual into one link ``thrust_frame`` whose origin is the centre of mass and
whose +Z is the model's +X (the face opposite the thruster cluster), so a body-frame force on +Z thrusts through
the centre of mass. Run from this directory: ``python make_turksat_orbit_urdf.py``.
"""

import xml.etree.ElementTree as ET

import numpy as np

SRC, DST = "spacecraft.urdf", "turksat_orbit.urdf"
# Thrust frame axes in model coordinates (columns): +Z_new = +X_model, +Y_new = +Y_model, +X_new = -Z_model.
R_NEW = np.array([[0.0, 0.0, 1.0], [0.0, 1.0, 0.0], [-1.0, 0.0, 0.0]])


def rpy_to_r(r, p, y):
    cr, sr, cp, sp, cy, sy = np.cos(r), np.sin(r), np.cos(p), np.sin(p), np.cos(y), np.sin(y)
    return np.array(
        [[cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
         [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
         [-sp, cp * sr, cp * cr]]
    )


def r_to_rpy(R):
    p = np.arcsin(-np.clip(R[2, 0], -1.0, 1.0))
    if abs(np.cos(p)) < 1e-9:  # gimbal lock: put everything in yaw
        return 0.0, p, np.arctan2(-R[0, 1], R[1, 1])
    return np.arctan2(R[2, 1], R[2, 2]), p, np.arctan2(R[1, 0], R[0, 0])


def origin(elem):
    o = elem.find("origin") if elem is not None else None
    xyz = [float(v) for v in (o.get("xyz", "0 0 0") if o is not None else "0 0 0").split()]
    rpy = [float(v) for v in (o.get("rpy", "0 0 0") if o is not None else "0 0 0").split()]
    return rpy_to_r(*rpy), np.array(xyz)


def main():
    root = ET.parse(SRC).getroot()
    links = {l.get("name"): l for l in root.findall("link")}
    parent_joint = {j.find("child").get("link"): j for j in root.findall("joint")}
    assert all(j.get("type") == "fixed" for j in root.findall("joint")), "expected only fixed joints"

    pose = {}

    def world(name):  # link pose in the model root frame
        if name not in pose:
            j = parent_joint.get(name)
            if j is None:
                pose[name] = (np.eye(3), np.zeros(3))
            else:
                Rp, pp = world(j.find("parent").get("link"))
                R, p = origin(j)
                pose[name] = (Rp @ R, pp + Rp @ p)
        return pose[name]

    # Composite mass, centre of mass and inertia (about the CoM, model axes).
    items = []
    for name, link in links.items():
        inert = link.find("inertial")
        m = float(inert.find("mass").get("value"))
        I = inert.find("inertia")
        Il = np.array(
            [[float(I.get("ixx")), float(I.get("ixy")), float(I.get("ixz"))],
             [float(I.get("ixy")), float(I.get("iyy")), float(I.get("iyz"))],
             [float(I.get("ixz")), float(I.get("iyz")), float(I.get("izz"))]]
        )
        Rw, pw = world(name)
        Ri, pi = origin(inert)
        items.append((m, pw + Rw @ pi, (Rw @ Ri) @ Il @ (Rw @ Ri).T))
    mass = sum(m for m, _, _ in items)
    com = sum(m * c for m, c, _ in items) / mass
    I_com = sum(Ic + m * (np.dot(c - com, c - com) * np.eye(3) - np.outer(c - com, c - com)) for m, c, Ic in items)
    I_new = R_NEW.T @ I_com @ R_NEW

    out = ET.Element("robot", name="turksat_orbit")
    link = ET.SubElement(out, "link", name="thrust_frame")
    inert = ET.SubElement(link, "inertial")
    ET.SubElement(inert, "origin", xyz="0 0 0", rpy="0 0 0")
    ET.SubElement(inert, "mass", value=f"{mass:.6f}")
    ET.SubElement(
        inert, "inertia",
        ixx=f"{I_new[0, 0]:.6f}", ixy=f"{I_new[0, 1]:.6f}", ixz=f"{I_new[0, 2]:.6f}",
        iyy=f"{I_new[1, 1]:.6f}", iyz=f"{I_new[1, 2]:.6f}", izz=f"{I_new[2, 2]:.6f}",
    )
    n_vis = 0
    for name, src_link in links.items():
        Rw, pw = world(name)
        for vis in src_link.findall("visual"):
            Rv, pv = origin(vis)
            R = R_NEW.T @ (Rw @ Rv)
            p = R_NEW.T @ (pw + Rw @ pv - com)
            v = ET.SubElement(link, "visual")
            ET.SubElement(v, "origin", xyz=" ".join(f"{x:.6f}" for x in p), rpy=" ".join(f"{x:.6f}" for x in r_to_rpy(R)))
            v.append(vis.find("geometry"))
            if vis.find("material") is not None:
                v.append(vis.find("material"))
            n_vis += 1
    # Isaac Lab only builds an Articulation from a URDF with a joint: hang a 1 g marker on a fixed joint at the CoM.
    tip = ET.SubElement(out, "link", name="com_marker")
    ti = ET.SubElement(tip, "inertial")
    ET.SubElement(ti, "origin", xyz="0 0 0", rpy="0 0 0")
    ET.SubElement(ti, "mass", value="0.001")
    ET.SubElement(ti, "inertia", ixx="1e-7", ixy="0", ixz="0", iyy="1e-7", iyz="0", izz="1e-7")
    j = ET.SubElement(out, "joint", name="com_marker_fixed", type="fixed")
    ET.SubElement(j, "origin", xyz="0 0 0", rpy="0 0 0")
    ET.SubElement(j, "parent", link="thrust_frame")
    ET.SubElement(j, "child", link="com_marker")
    ET.indent(out)
    ET.ElementTree(out).write(DST, xml_declaration=True, encoding="utf-8")
    print(f"wrote {DST}: {n_vis} visuals, mass {mass:.3f} kg, CoM (model frame) {np.round(com, 4)}")
    print("inertia about CoM, thrust frame:\n", np.round(I_new, 3))


if __name__ == "__main__":
    main()
