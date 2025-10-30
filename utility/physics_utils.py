# Copyright 2025 ZTE Corporation.
# All Rights Reserved.
#
#    Licensed under the Apache License, Version 2.0 (the "License"); you may
#    not use this file except in compliance with the License. You may obtain
#    a copy of the License at
#
#         http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
#    WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
#    License for the specific language governing permissions and limitations
#    under the License.

"""
Physics Utilities

Provides utility functions for diagnosing and fixing common USD physics issues,
particularly related to rigid body hierarchies and joint configurations.
"""

from typing import List, Tuple, Optional
import omni.usd
from pxr import Usd, UsdPhysics, UsdGeom, Gf
from utility.logger import Logger


class PhysicsDiagnostics:
    """Diagnostic tools for USD physics configuration issues."""
    
    @staticmethod
    def find_nested_rigid_bodies(stage: Usd.Stage, root_path: str = "/") -> List[Tuple[str, str]]:
        """Find nested rigid body hierarchies that may cause simulation issues.
        
        Args:
            stage: USD stage to search.
            root_path: Root path to start search from.
            
        Returns:
            List of (parent_path, child_path) tuples where both have RigidBodyAPI.
        """
        nested_bodies = []
        
        def traverse_prim(prim: Usd.Prim, parent_has_rigidbody: bool = False):
            if not prim.IsValid():
                return
            
            prim_path = str(prim.GetPath())
            has_rigidbody = prim.HasAPI(UsdPhysics.RigidBodyAPI)
            
            # If both parent and current prim have rigid body, it's a nested hierarchy
            if parent_has_rigidbody and has_rigidbody:
                # Find the parent with rigid body
                parent = prim.GetParent()
                while parent.IsValid() and not parent.HasAPI(UsdPhysics.RigidBodyAPI):
                    parent = parent.GetParent()
                if parent.IsValid():
                    nested_bodies.append((str(parent.GetPath()), prim_path))
            
            # Recurse to children
            for child in prim.GetChildren():
                traverse_prim(child, has_rigidbody or parent_has_rigidbody)
        
        root_prim = stage.GetPrimAtPath(root_path)
        if root_prim.IsValid():
            traverse_prim(root_prim)
        
        return nested_bodies
    
    @staticmethod
    def find_invalid_joints(stage: Usd.Stage, root_path: str = "/") -> List[str]:
        """Find joints with missing or invalid body references.
        
        Args:
            stage: USD stage to search.
            root_path: Root path to start search from.
            
        Returns:
            List of joint prim paths with issues.
        """
        invalid_joints = []
        
        def traverse_prim(prim: Usd.Prim):
            if not prim.IsValid():
                return
            
            # Check if prim is a joint
            if prim.IsA(UsdPhysics.Joint):
                joint = UsdPhysics.Joint(prim)
                body0_rel = joint.GetBody0Rel()
                body1_rel = joint.GetBody1Rel()
                
                # Check if bodies are defined
                has_body0 = body0_rel and len(body0_rel.GetTargets()) > 0
                has_body1 = body1_rel and len(body1_rel.GetTargets()) > 0
                
                if not has_body0 or not has_body1:
                    invalid_joints.append(str(prim.GetPath()))
            
            # Recurse to children
            for child in prim.GetChildren():
                traverse_prim(child)
        
        root_prim = stage.GetPrimAtPath(root_path)
        if root_prim.IsValid():
            traverse_prim(root_prim)
        
        return invalid_joints
    
    @staticmethod
    def find_non_uniform_scales(stage: Usd.Stage, root_path: str = "/", tolerance: float = 0.01) -> List[str]:
        """Find rigid bodies with non-uniform scale (can cause physics issues).
        
        Args:
            stage: USD stage to search.
            root_path: Root path to start search from.
            tolerance: Tolerance for considering scale uniform (default 1%).
            
        Returns:
            List of prim paths with non-uniform scale.
        """
        non_uniform_prims = []
        
        def traverse_prim(prim: Usd.Prim):
            if not prim.IsValid():
                return
            
            if prim.HasAPI(UsdPhysics.RigidBodyAPI):
                xformable = UsdGeom.Xformable(prim)
                if xformable:
                    # Get local transform
                    local_xform = xformable.GetLocalTransformation()
                    
                    # Extract scale from transform matrix
                    scale_x = (local_xform.GetRow(0).GetLength())
                    scale_y = (local_xform.GetRow(1).GetLength())
                    scale_z = (local_xform.GetRow(2).GetLength())
                    
                    # Check if scale is uniform
                    max_scale = max(scale_x, scale_y, scale_z)
                    min_scale = min(scale_x, scale_y, scale_z)
                    
                    if max_scale > 0 and abs(max_scale - min_scale) / max_scale > tolerance:
                        non_uniform_prims.append(str(prim.GetPath()))
            
            # Recurse to children
            for child in prim.GetChildren():
                traverse_prim(child)
        
        root_prim = stage.GetPrimAtPath(root_path)
        if root_prim.IsValid():
            traverse_prim(root_prim)
        
        return non_uniform_prims
    
    @staticmethod
    def print_diagnostics_report(stage: Usd.Stage, scene_path: str = "/scene"):
        """Print comprehensive physics diagnostics report.
        
        Args:
            stage: USD stage to diagnose.
            scene_path: Path to scene root (default "/scene").
        """
        Logger.info("\n" + "=" * 70)
        Logger.info("PHYSICS DIAGNOSTICS REPORT")
        Logger.info("=" * 70)
        
        # Find nested rigid bodies
        nested = PhysicsDiagnostics.find_nested_rigid_bodies(stage, scene_path)
        if nested:
            Logger.warning(f"\n⚠️  Found {len(nested)} nested rigid body hierarchies:")
            for parent, child in nested:
                Logger.warning(f"   Parent: {parent}")
                Logger.warning(f"   Child:  {child}")
                Logger.warning("")
        else:
            Logger.info("✓ No nested rigid body hierarchies found")
        
        # Find invalid joints
        invalid_joints = PhysicsDiagnostics.find_invalid_joints(stage, scene_path)
        if invalid_joints:
            Logger.warning(f"\n⚠️  Found {len(invalid_joints)} joints with missing body references:")
            for joint_path in invalid_joints:
                Logger.warning(f"   {joint_path}")
        else:
            Logger.info("✓ All joints have valid body references")
        
        # Find non-uniform scales
        non_uniform = PhysicsDiagnostics.find_non_uniform_scales(stage, scene_path)
        if non_uniform:
            Logger.warning(f"\n⚠️  Found {len(non_uniform)} rigid bodies with non-uniform scale:")
            for prim_path in non_uniform:
                Logger.warning(f"   {prim_path}")
        else:
            Logger.info("✓ All rigid bodies have uniform scale")
        
        Logger.info("=" * 70 + "\n")


class PhysicsFixer:
    """Tools to automatically fix common physics configuration issues."""
    
    @staticmethod
    def disable_child_rigid_bodies(stage: Usd.Stage, parent_path: str) -> int:
        """Disable RigidBodyAPI on all children of a parent rigid body.
        
        This fixes the nested rigid body hierarchy issue by disabling physics
        on child prims.
        
        Args:
            stage: USD stage.
            parent_path: Path to parent prim with RigidBodyAPI.
            
        Returns:
            Number of child rigid bodies disabled.
        """
        parent_prim = stage.GetPrimAtPath(parent_path)
        if not parent_prim.IsValid() or not parent_prim.HasAPI(UsdPhysics.RigidBodyAPI):
            Logger.warning(f"Invalid parent or missing RigidBodyAPI: {parent_path}")
            return 0
        
        disabled_count = 0
        
        def disable_children(prim: Usd.Prim):
            nonlocal disabled_count
            for child in prim.GetChildren():
                if child.HasAPI(UsdPhysics.RigidBodyAPI):
                    # Remove RigidBodyAPI
                    child.RemoveAPI(UsdPhysics.RigidBodyAPI)
                    disabled_count += 1
                    Logger.info(f"Disabled RigidBodyAPI on: {child.GetPath()}")
                
                # Recurse to grandchildren
                disable_children(child)
        
        disable_children(parent_prim)
        return disabled_count
    
    @staticmethod
    def remove_invalid_joints(stage: Usd.Stage, scene_path: str = "/scene") -> int:
        """Remove joints with missing body references.
        
        Args:
            stage: USD stage.
            scene_path: Path to scene root.
            
        Returns:
            Number of joints removed.
        """
        invalid_joints = PhysicsDiagnostics.find_invalid_joints(stage, scene_path)
        
        for joint_path in invalid_joints:
            prim = stage.GetPrimAtPath(joint_path)
            if prim.IsValid():
                stage.RemovePrim(joint_path)
                Logger.info(f"Removed invalid joint: {joint_path}")
        
        return len(invalid_joints)
    
    @staticmethod
    def fix_nested_rigid_bodies(stage: Usd.Stage, scene_path: str = "/scene") -> int:
        """Automatically fix nested rigid body hierarchies.
        
        Args:
            stage: USD stage.
            scene_path: Path to scene root.
            
        Returns:
            Total number of fixes applied.
        """
        nested = PhysicsDiagnostics.find_nested_rigid_bodies(stage, scene_path)
        total_fixed = 0
        
        # Group by parent
        parent_to_children = {}
        for parent, child in nested:
            if parent not in parent_to_children:
                parent_to_children[parent] = []
            parent_to_children[parent].append(child)
        
        # Disable children for each parent
        for parent_path, children in parent_to_children.items():
            Logger.info(f"Fixing nested rigid bodies under: {parent_path}")
            for child_path in children:
                child_prim = stage.GetPrimAtPath(child_path)
                if child_prim.IsValid() and child_prim.HasAPI(UsdPhysics.RigidBodyAPI):
                    child_prim.RemoveAPI(UsdPhysics.RigidBodyAPI)
                    total_fixed += 1
                    Logger.info(f"  Disabled RigidBodyAPI on child: {child_path}")
        
        return total_fixed


def diagnose_scene_physics(world, scene_path: str = "/scene", auto_fix: bool = False):
    """Convenience function to diagnose and optionally fix scene physics issues.
    
    Args:
        world: Isaac Sim World instance.
        scene_path: Path to scene root in USD.
        auto_fix: If True, automatically fix detected issues.
        
    Example:
        >>> from utility.physics_utils import diagnose_scene_physics
        >>> diagnose_scene_physics(world, scene_path="/scene", auto_fix=True)
    """
    stage = world.stage
    
    # Print diagnostics
    PhysicsDiagnostics.print_diagnostics_report(stage, scene_path)
    
    if auto_fix:
        Logger.info("\n🔧 Applying automatic fixes...")
        
        # Fix nested rigid bodies
        fixed_nested = PhysicsFixer.fix_nested_rigid_bodies(stage, scene_path)
        if fixed_nested > 0:
            Logger.info(f"✓ Fixed {fixed_nested} nested rigid body issues")
        
        # Remove invalid joints
        removed_joints = PhysicsFixer.remove_invalid_joints(stage, scene_path)
        if removed_joints > 0:
            Logger.info(f"✓ Removed {removed_joints} invalid joints")
        
        if fixed_nested == 0 and removed_joints == 0:
            Logger.info("✓ No fixes needed")
        
        Logger.info("")
