"""Procedural PS1-style low-poly humanoid actor.

Built from hierarchical box primitives to mimic the crunchy, low-poly
look of PS1-era 3D (Tomb Raider, Metal Gear Solid).  No external model
files required — everything is generated at runtime.
"""

import math

from panda3d.core import (
    GeomNode, Geom, GeomVertexData, GeomVertexFormat, GeomVertexWriter,
    GeomTriangles, NodePath,
)


def _make_box(name, w, h, d, color):
    """Build a single box primitive as a NodePath."""
    fmt = GeomVertexFormat.getV3n3c4()
    vdata = GeomVertexData(name, fmt, Geom.UHStatic)
    vwrite = GeomVertexWriter(vdata, "vertex")
    nwrite = GeomVertexWriter(vdata, "normal")
    cwrite = GeomVertexWriter(vdata, "color")

    x, y, z = w * 0.5, h * 0.5, d * 0.5

    faces = [
        ((0, 0, 1),  [(-x, -y, z), (x, -y, z), (x, y, z), (-x, y, z)]),
        ((0, 0, -1), [(-x, y, -z), (x, y, -z), (x, -y, -z), (-x, -y, -z)]),
        ((1, 0, 0),  [(x, -y, -z), (x, y, -z), (x, y, z), (x, -y, z)]),
        ((-1, 0, 0), [(-x, -y, z), (-x, y, z), (-x, y, -z), (-x, -y, -z)]),
        ((0, 1, 0),  [(-x, y, -z), (-x, y, z), (x, y, z), (x, y, -z)]),
        ((0, -1, 0), [(x, -y, -z), (x, -y, z), (-x, -y, z), (-x, -y, -z)]),
    ]

    tri = GeomTriangles(Geom.UHStatic)
    vi = 0
    for normal, corners in faces:
        for corner in corners:
            vwrite.addData3(*corner)
            nwrite.addData3(*normal)
            cwrite.addData4(*color)
        tri.addVertices(vi, vi + 1, vi + 2)
        tri.addVertices(vi, vi + 2, vi + 3)
        vi += 4

    geom = Geom(vdata)
    geom.addPrimitive(tri)
    node = GeomNode(name)
    node.addGeom(geom)
    return NodePath(node)


class PS1Humanoid(object):
    """Hierarchical low-poly humanoid with poseable joints.

    Skeleton (all NodePaths, parented hierarchically)::

        root
        └── hips
            ├── spine
            │   └── chest
            │       ├── head
            │       ├── left_shoulder
            │       │   └── left_upper_arm
            │       │       └── left_lower_arm
            │       └── right_shoulder
            │           └── right_upper_arm
            │               └── right_lower_arm
            ├── left_hip_joint
            │   └── left_upper_leg
            │       └── left_lower_leg
            └── right_hip_joint
                └── right_upper_leg
                    └── right_lower_leg
    """

    TORSO_W, TORSO_H, TORSO_D = 0.50, 0.70, 0.30
    HEAD_W, HEAD_H, HEAD_D = 0.35, 0.35, 0.35
    ARM_W, ARM_H, ARM_D = 0.12, 0.45, 0.12
    LEG_W, LEG_H, LEG_D = 0.16, 0.55, 0.16

    def __init__(self, parent):
        # type: (NodePath) -> None
        self.root = parent.attachNewNode("humanoid-root")

        # hips (root of body)
        self.hips = self.root.attachNewNode("hips")
        self.hips.setPos(0, 0, 1.0)

        # spine / torso
        self.spine = self.hips.attachNewNode("spine")
        self.torso = _make_box(
            "torso", self.TORSO_W, self.TORSO_H, self.TORSO_D,
            (0.8, 0.2, 0.2, 1.0))
        self.torso.reparentTo(self.spine)
        self.torso.setPos(0, 0, self.TORSO_H * 0.5)

        # chest (upper body pivot)
        self.chest = self.spine.attachNewNode("chest")
        self.chest.setPos(0, 0, self.TORSO_H * 0.8)

        # head
        self.head_joint = self.chest.attachNewNode("head-joint")
        self.head_joint.setPos(0, 0, 0.05)
        self.head = _make_box(
            "head", self.HEAD_W, self.HEAD_H, self.HEAD_D,
            (0.9, 0.7, 0.5, 1.0))
        self.head.reparentTo(self.head_joint)
        self.head.setPos(0, 0, self.HEAD_H * 0.5)

        # arms
        self.left_shoulder = self.chest.attachNewNode("left-shoulder")
        self.left_shoulder.setPos(self.TORSO_W * 0.55, 0, 0)
        self.left_upper_arm = _make_box(
            "left-upper-arm", self.ARM_W, self.ARM_H, self.ARM_D,
            (0.2, 0.5, 0.8, 1.0))
        self.left_upper_arm.reparentTo(self.left_shoulder)
        self.left_upper_arm.setPos(0, 0, -self.ARM_H * 0.5)
        self.left_elbow = self.left_upper_arm.attachNewNode("left-elbow")
        self.left_elbow.setPos(0, 0, -self.ARM_H * 0.5)
        self.left_lower_arm = _make_box(
            "left-lower-arm", self.ARM_W * 0.85, self.ARM_H, self.ARM_D * 0.85,
            (0.2, 0.5, 0.8, 1.0))
        self.left_lower_arm.reparentTo(self.left_elbow)
        self.left_lower_arm.setPos(0, 0, -self.ARM_H * 0.5)

        self.right_shoulder = self.chest.attachNewNode("right-shoulder")
        self.right_shoulder.setPos(-self.TORSO_W * 0.55, 0, 0)
        self.right_upper_arm = _make_box(
            "right-upper-arm", self.ARM_W, self.ARM_H, self.ARM_D,
            (0.2, 0.5, 0.8, 1.0))
        self.right_upper_arm.reparentTo(self.right_shoulder)
        self.right_upper_arm.setPos(0, 0, -self.ARM_H * 0.5)
        self.right_elbow = self.right_upper_arm.attachNewNode("right-elbow")
        self.right_elbow.setPos(0, 0, -self.ARM_H * 0.5)
        self.right_lower_arm = _make_box(
            "right-lower-arm", self.ARM_W * 0.85, self.ARM_H, self.ARM_D * 0.85,
            (0.2, 0.5, 0.8, 1.0))
        self.right_lower_arm.reparentTo(self.right_elbow)
        self.right_lower_arm.setPos(0, 0, -self.ARM_H * 0.5)

        # legs
        self.left_hip = self.hips.attachNewNode("left-hip")
        self.left_hip.setPos(self.TORSO_W * 0.3, 0, 0)
        self.left_upper_leg = _make_box(
            "left-upper-leg", self.LEG_W, self.LEG_H, self.LEG_D,
            (0.3, 0.3, 0.3, 1.0))
        self.left_upper_leg.reparentTo(self.left_hip)
        self.left_upper_leg.setPos(0, 0, -self.LEG_H * 0.5)
        self.left_knee = self.left_upper_leg.attachNewNode("left-knee")
        self.left_knee.setPos(0, 0, -self.LEG_H * 0.5)
        self.left_lower_leg = _make_box(
            "left-lower-leg", self.LEG_W * 0.85, self.LEG_H, self.LEG_D * 0.85,
            (0.3, 0.3, 0.3, 1.0))
        self.left_lower_leg.reparentTo(self.left_knee)
        self.left_lower_leg.setPos(0, 0, -self.LEG_H * 0.5)

        self.right_hip = self.hips.attachNewNode("right-hip")
        self.right_hip.setPos(-self.TORSO_W * 0.3, 0, 0)
        self.right_upper_leg = _make_box(
            "right-upper-leg", self.LEG_W, self.LEG_H, self.LEG_D,
            (0.3, 0.3, 0.3, 1.0))
        self.right_upper_leg.reparentTo(self.right_hip)
        self.right_upper_leg.setPos(0, 0, -self.LEG_H * 0.5)
        self.right_knee = self.right_upper_leg.attachNewNode("right-knee")
        self.right_knee.setPos(0, 0, -self.LEG_H * 0.5)
        self.right_lower_leg = _make_box(
            "right-lower-leg", self.LEG_W * 0.85, self.LEG_H, self.LEG_D * 0.85,
            (0.3, 0.3, 0.3, 1.0))
        self.right_lower_leg.reparentTo(self.right_knee)
        self.right_lower_leg.setPos(0, 0, -self.LEG_H * 0.5)

        # Flat name -> NodePath map for animation sampling
        self.joints = {
            "hips": self.hips,
            "spine": self.spine,
            "chest": self.chest,
            "head": self.head_joint,
            "left_shoulder": self.left_shoulder,
            "left_elbow": self.left_elbow,
            "right_shoulder": self.right_shoulder,
            "right_elbow": self.right_elbow,
            "left_hip": self.left_hip,
            "left_knee": self.left_knee,
            "right_hip": self.right_hip,
            "right_knee": self.right_knee,
        }

    def apply_pose(self, pose):
        # type: (dict) -> None
        """Apply a pose dict.

        Each entry is either::

            {"hpr": (h, p, r)}
            or
            {"hpr": (h, p, r), "pos": (x, y, z)}

        Missing joints are left as-is.
        """
        for name, data in pose.items():
            node = self.joints.get(name)
            if node is None:
                continue
            hpr = data.get("hpr")
            if hpr is not None:
                node.setHpr(float(hpr[0]), float(hpr[1]), float(hpr[2]))
            pos = data.get("pos")
            if pos is not None:
                node.setPos(float(pos[0]), float(pos[1]), float(pos[2]))
