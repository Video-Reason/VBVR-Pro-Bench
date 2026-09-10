"""Human-facing semantics for every registered VBVR task.

Scoring stays in the evaluator/adapters.  This table only controls concise,
task-specific wording shown in the HUD.
"""

from __future__ import annotations


def _s(title, failure, failures=None, **labels):
    return {"title": title, "failure": failure,
            "failures": failures or {}, "labels": labels}


TASK_SEMANTICS = {
    "G-131_select_next_figure_increasing_size_sequence_data-generator": _s("Increasing-size sequence", "Wrong next figure selected", match_score="Next-figure selection", selection_target="next figure", selection_wrong="Wrong sequence figure selected", selection_missed="Next figure not selected"),
    "G-134_select_next_figure_large_small_alternating_sequence_data-generator": _s("Large-small sequence", "Alternating-size choice is wrong", match_score="Next-figure selection", selection_target="alternating-size figure", selection_wrong="Wrong alternating-size figure selected", selection_missed="Next alternating figure not selected"),
    "G-135_select_next_figure_small_large_alternating_sequence_data-generator": _s("Small-large sequence", "Alternating-size choice is wrong", match_score="Next-figure selection", selection_target="alternating-size figure", selection_wrong="Wrong alternating-size figure selected", selection_missed="Next alternating figure not selected"),
    "G-136_locate_point_in_overlapping_area_data-generator": _s("Locate overlap", "Overlap point is missing or misplaced", match_score="Overlap-point accuracy", selection_target="overlap point", selection_wrong="Point marked outside the overlap", selection_missed="Overlap point not marked"),
    "G-138_spot_unique_non_repeated_color_data-generator": _s("Unique colour", "Unique-colour object was not identified", primary="Identification accuracy"),
    "G-13_grid_number_sequence_data-generator": _s("Grid number sequence", "Number route is incomplete or out of order", task_score="Sequence-route accuracy"),
    "G-140_locate_topmost_unobscured_figure_data-generator": _s("Topmost visible figure", "Wrong visible figure identified", primary="Identification accuracy"),
    "G-147_identify_unique_figure_in_uniform_set_data-generator": _s("Unique figure", "Wrong figure selected", match_score="Unique-figure selection", selection_target="unique figure", selection_wrong="Non-unique figure selected", selection_missed="Unique figure not selected"),
    "G-158_identify_all_hollow_points_data-generator": _s("All hollow points", "Hollow-point selection is incorrect", match_score="Hollow-point selection"),
    "G-15_grid_avoid_obstacles_data-generator": _s("Obstacle-avoiding route", "Route missed the goal or hit an obstacle", task_score="Safe-route quality"),
    "G-160_circle_largest_numerical_value_data-generator": _s("Largest value", "Largest numerical value was not circled", match_score="Largest-value selection", selection_target="largest value", selection_wrong="Non-largest value circled", selection_missed="Largest value not circled"),
    "G-161_mark_second_largest_shape_data-generator": _s("Second-largest shape", "Second-largest shape was marked incorrectly", match_score="Target selection", shape_score="Mark shape quality"),
    "G-167_select_longest_polygon_side_data-generator": _s("Longest polygon side", "Longest side was not selected", match_score="Longest-side selection", selection_target="longest side", selection_wrong="Non-longest side selected", selection_missed="Longest side not selected"),
    "G-168_identify_nearest_to_square_rectangle_data-generator": _s("Most square-like rectangle", "Wrong rectangle selected", match_score="Rectangle selection", selection_target="most square-like rectangle", selection_wrong="Less square-like rectangle selected", selection_missed="Most square-like rectangle not selected"),
    "G-169_locate_intersection_of_segments_data-generator": _s("Segment intersection", "Intersection point is missing or misplaced", match_score="Intersection accuracy", selection_target="intersection point", selection_wrong="Point marked away from the intersection", selection_missed="Intersection point not marked"),
    "G-16_grid_go_through_block_data-generator": _s("Required-block route", "Route missed a required block", task_score="Required-block route"),
    "G-174_arrange_circles_by_circumference_data-generator": _s("Sort circles by circumference", "Circle order is incorrect", primary="Ordering accuracy"),
    "G-189_draw_midpoint_perpendicular_line_data-generator": _s("Perpendicular bisector", "Required perpendicular line is inaccurate", core="Line accuracy"),
    "G-18_grid_shortest_path_data-generator": _s("Shortest grid route", "Route is incomplete or not shortest", task_score="Shortest-path quality"),
    "G-193_draw_next_sized_shape_data-generator": _s("Next-sized shape", "Expected next-sized shape is missing", completion="Shape completion"),
    "G-194_construct_concentric_ring_data-generator": _s("Concentric rings", "Ring arrangement is inaccurate", primary="Ring arrangement"),
    "G-202_mark_wave_peaks_data-generator": _s("Wave peaks", "One or more wave peaks were marked incorrectly", match_score="Peak-marking accuracy"),
    "G-206_identify_pentagons_data-generator": _s("Identify pentagons", "Pentagon selection is incorrect", match_score="Pentagon selection", selection_target="pentagons", selection_wrong="Non-pentagon selected", selection_missed="One or more pentagons missed"),
    "G-212_find_incorrect_arrow_direction_data-generator": _s("Incorrect arrow", "Wrong arrow was selected", match_score="Arrow selection", selection_target="wrong-direction arrow", selection_wrong="Correct-direction arrow selected", selection_missed="Wrong-direction arrow not selected"),
    "G-217_circle_central_dot_data-generator": _s("Central dot", "Central dot was not circled correctly", match_score="Central-dot selection", selection_target="central dot", selection_wrong="Non-central dot circled", selection_missed="Central dot not circled"),
    "G-218_identify_largest_angle_in_triangle_data-generator": _s("Largest triangle angle", "Largest angle was not identified", match_score="Largest-angle selection", selection_target="largest angle", selection_wrong="Non-largest angle marked", selection_missed="Largest angle not marked"),
    "G-219_select_leftmost_shape_data-generator": _s("Leftmost shape", "Wrong shape selected", match_score="Leftmost-shape selection", selection_target="leftmost shape", selection_wrong="Non-leftmost shape selected", selection_missed="Leftmost shape not selected"),
    "G-21_multiple_occlusions_vertical_data-generator": _s("Vertical occlusion", "Mask motion or occlusion sequence is incorrect", failures={"occlusion_correctness": "Occlusion order is incorrect", "elements_preservation": "Scene elements changed"}, mask_path_vadility="Mask trajectory", occlusion_correctness="Occlusion sequence", elements_preservation="Scene preservation"),
    "G-221_outline_innermost_square_data-generator": _s("Innermost square", "Innermost square was not outlined correctly", primary="Outline accuracy"),
    "G-222_mark_tangent_point_of_circles_data-generator": _s("Circle tangent point", "Tangent point is missing or misplaced", match_score="Tangent-point accuracy", selection_target="tangent point", selection_wrong="Point marked away from tangency", selection_missed="Tangent point not marked"),
    "G-223_highlight_horizontal_lines_data-generator": _s("Horizontal lines", "Horizontal-line selection is incorrect", match_score="Line selection", selection_target="horizontal lines", selection_wrong="Non-horizontal line highlighted", selection_missed="One or more horizontal lines missed"),
    "G-240_add_borders_to_unbordered_shapes_data-generator": _s("Add missing borders", "Border edits are incorrect", primary="Border-edit accuracy"),
    "G-247_identify_chinese_character_data-generator": _s("Chinese character", "Wrong character selected", match_score="Character selection", selection_target="Chinese character", selection_wrong="Non-Chinese symbol selected", selection_missed="Chinese character not selected"),
    "G-248_mark_asymmetrical_shape_data-generator": _s("Asymmetrical shape", "Wrong shape selected", match_score="Asymmetry selection", selection_target="asymmetrical shape", selection_wrong="Symmetrical shape selected", selection_missed="Asymmetrical shape not selected"),
    "G-24_separate_objects_no_spin_data-generator": _s("Separate objects", "Objects were not separated by pure translation", failures={"alignment_gate": "Poor final separation gated motion credit", "non_alignment_score": "Translation path is invalid"}, alignment="Final separation", non_alignment_score="Translation process"),
    "G-250_color_triple_intersection_red_data-generator": _s("Triple intersection", "Triple-overlap colouring is incorrect", core="Red-region accuracy"),
    "G-25_seperate_object_spinning_data-generator": _s("Separate and spin objects", "Required separation or rotation is inaccurate", failures={"alignment_gate": "Poor final separation gated motion credit", "non_alignment_score": "Translation or rotation process is invalid"}, alignment="Final separation", non_alignment_score="Motion and rotation"),
    "G-273_high_density_liquid_data-generator": _s("High-density liquid", "Liquid ordering or settling process is incorrect", final_state="Final liquid state", process="Settling process"),
    "G-29_chart_extreme_with_data_data-generator": _s("Chart extreme", "Wrong chart extreme was identified", primary="Extreme-value accuracy"),
    "G-31_directed_graph_navigation_data-generator": _s("Directed graph route", "Directed graph route is invalid", path_validity="Directed-path validity"),
    "G-39_attention_shift_different_data-generator": _s("Attention shift", "Attention box moved to the wrong object", box_position="Box position"),
    "G-3_stable_sort_data-generator": _s("Stable sort", "Objects are not in stable sorted order", primary="Stable-sort accuracy"),
    "G-41_grid_highest_cost_data-generator": _s("Highest-cost grid route", "Route does not collect the required maximum cost", task_score="Route cost accuracy"),
    "G-43_understand_scene_structure_data-generator": _s("Scene structure", "Scene relationship was identified incorrectly", primary="Structure accuracy"),
    "G-45_key_door_matching_data-generator": _s("Key-door navigation", "Correct key or door route was not completed"),
    "G-47_multiple_keys_for_one_door_data-generator": _s("Multiple-key navigation", "Required keys were missed before the door", task_score="Key-route accuracy"),
    "G-51_predict_next_color_data-generator": _s("Predict next colour", "Predicted colour is incorrect", completion="Colour prediction"),
    "G-54_connecting_color_data-generator": _s("Connect matching colours", "Colour connections are missing or incorrect", core="Correct connections"),
    "G-5_multi_object_placement_data-generator": _s("Multi-object placement", "Object path or endpoint is incorrect"),
    "G-8_track_object_movement_data-generator": _s("Move marked object", "Marked object did not reach the target correctly"),
    "G-9_identify_objects_in_region_data-generator": _s("Objects in region", "Region-object selection is incorrect", primary="Region identification"),
    "O-10_shape_outline_fill_data-generator": _s("Outline then fill", "Required outline/fill result is missing", completion="Transformation completion"),
    "O-11_shape_color_then_move_data-generator": _s("Colour then move", "Required recolouring or movement is incomplete", completion="Transformation completion"),
    "O-12_shape_color_then_scale_data-generator": _s("Colour then scale", "Required recolouring or scaling is incomplete", completion="Transformation completion"),
    "O-13_shape_outline_then_move_data-generator": _s("Outline then move", "Required outline or movement is incomplete", completion="Transformation completion"),
    "O-14_shape_scale_then_outline_data-generator": _s("Scale then outline", "Scaled outline is missing", completion="Transformation completion"),
    "O-15_ball_bounces_given_time_data-generator": _s("Timed ball bounces", "Bounce motion is incorrect", failures={"trajectory_coverage": "Bounce path coverage is incomplete", "trajectory": "Bounce trajectory does not match", "foreground_deduction": "Foreground was damaged"}, physics="Bounce physics", trajectory="Trajectory match"),
    "O-16_color_addition_data-generator": _s("Additive colour mixing", "Colour mixing result is incorrect", mixing_color="Mixed colour"),
    "O-18_glass_refraction_data-generator": _s("Glass refraction", "Refracted ray is inaccurate", core="Refracted-ray accuracy"),
    "O-19_mirror_reflection_data-generator": _s("Mirror reflection", "Reflected ray is inaccurate", core="Reflected-ray accuracy"),
    "O-21_construction_blueprint_data-generator": _s("Construction blueprint", "Blueprint option or colour coding is incorrect", failures={"correct_option_green": "Correct option is not green", "other_options_red": "Incorrect options are not red"}, shape_matching="Blueprint match", correct_option_green="Correct option marked green", other_options_red="Other options marked red"),
    "O-22_construction_stack_data-generator": _s("Stack construction", "Final stack or construction order is incorrect", failures={"process_gate": "Invalid assembly sequence"}, main_score="Final stack", process_gate="Assembly process gate"),
    "O-23_domino_chain_branch_path_prediction_data-generator": _s("Domino branch prediction", "Wrong branch fell or fall sequence is invalid", failures={"process": "Domino fall order is invalid"}, final_state="Final domino state", process="Falling process"),
    "O-24_domino_chain_gap_analysis_data-generator": _s("Domino gap analysis", "Domino gap behaviour is incorrect", failures={"process": "Domino propagation is invalid"}, final_state="Final domino state", process="Falling process"),
    "O-25_LEGO_construction_assembly_data-generator": _s("LEGO assembly", "Assembly structure is incomplete or incorrect", assembly_correctness="Assembly correctness"),
    "O-27_move_2_object_to_2_target_data-generator": _s("Two-object placement", "An object missed its target or moved out of sync", movement="Object movement"),
    "O-29_ballcolor_data-generator": _s("Merge coloured balls", "Ball merge sequence or final label is incorrect", final_state="Final merged cluster", merge_process="Merge sequence"),
    "O-2_pigment_color_mixing_subtractive_data-generator": _s("Subtractive pigment mixing", "Pigment mixing result is incorrect", failures={"object_preservation": "Source pigment objects changed"}, mixing_color="Mixed pigment colour"),
    "O-30_bookshelf_data-generator": _s("Bookshelf insertion", "Book placement or order is incorrect", failures={"sequential_insertion": "Book insertion order is incorrect"}, final_placement="Final book placement"),
    "O-31_ball_eating_data-generator": _s("Ball eating", "One or more balls were not absorbed correctly", final_state="Final eater state", eat_process="Absorption sequence"),
    "O-32_rolling_ball_data-generator": _s("Rolling ball", "Ball motion is incorrect", completion="Rolling completion"),
    "O-33_counting_object_data-generator": _s("Count objects", "Displayed count is incorrect", core="Count correctness"),
    "O-34_dot_to_dot_task_data-generator": _s("Dot to dot", "Dot connections are incorrect", failures={"order": "Dots were connected out of order", "numerical": "Dot numbers changed"}, completeness="Connection completeness"),
    "O-36_grid_shift_data-generator": _s("Grid shift", "Cells did not shift in the required sequence", failures={"process_score": "Cell-shift sequence is invalid", "final_cell_gate": "Incorrect final grid gated process credit"}, final_cell_score="Final grid layout", process_score="Shift process"),
    "O-37_light_sequence_data-generator": _s("Light sequence", "Wrong lights or order were shown", completion="Light-sequence accuracy"),
    "O-38_majority_color_data-generator": _s("Majority colour", "Wrong colour group was retained", completion="Majority-colour result"),
    "O-39_maze_data-generator": _s("Maze navigation", "Path missed the goal or crossed a wall", failures={"coverage": "Maze route is incomplete", "wall_multiplier": "Maze wall was crossed"}, proximity="Path proximity"),
    "O-43_object_subtraction_data-generator": _s("Object subtraction", "Wrong objects were removed or retained", completion="Subtraction result"),
    "O-44_rotation_puzzle_data-generator": _s("Rotation puzzle", "Final rotation or rotation sequence is incorrect", core="Final rotation state"),
    "O-45_sequence_completion_data-generator": _s("Sequence completion", "Generated object does not complete the sequence", core="Generated-object accuracy"),
    "O-46_shape_sorter_data-generator": _s("Shape sorter", "Shape sorting is incorrect", final_layout="Final sorted layout"),
    "O-47_sliding_puzzle_data-generator": _s("Sliding puzzle", "Final grid or move sequence is invalid", failures={"process": "Puzzle contains an illegal move"}, final_state="Final puzzle grid", process="Legal move sequence"),
    "O-49_symmetry_completion_data-generator": _s("Symmetry completion", "Missing cells do not complete the symmetry", completion="Symmetry completion"),
    "O-52_traffic_light_data-generator": _s("Traffic-light sequence", "Light states or countdown timing is incorrect", final_state="Final light state", process="Light-change sequence"),
    "O-53_clock_data-generator": _s("Clock motion", "Clock time or hand motion is incorrect", failures={"process_validity": "Clock hands moved incorrectly", "element_preservation": "Clock face changed"}, completion="Target time", process_validity="Hand-motion validity", element_preservation="Clock-face preservation"),
    "O-54_control_panel_data-generator": _s("Control panel", "Control-panel operation is incorrect", completion="Final control state"),
    "O-55_rotation_data-generator": _s("View rotation", "View rotation is incorrect", final_state="Final view", process="Rotation sequence"),
    "O-56_raven_data-generator": _s("Raven matrix", "Inserted answer does not complete the matrix", completion="Matrix completion"),
    "O-58_symbol_delete_data-generator": _s("Delete symbol", "Required symbol was not deleted cleanly", fg_score="Symbol deletion"),
    "O-59_symbol_insert_data-generator": _s("Insert symbol", "Symbol insertion is incorrect", failures={"template_score": "Template symbol changed"}, seq_score="Edited sequence", template_score="Template symbol preservation"),
    "O-5_symbol_deletion_data-generator": _s("Delete boxed symbol", "Symbol deletion is incorrect", delete_score="Target deletion"),
    "O-60_symbol_substitute_data-generator": _s("Substitute symbol", "Symbol substitution is incorrect", seq_score="Edited sequence"),
    "O-61_symbol_edit_data-generator": _s("Edit symbol sequence", "Edited sequence violates the requested operation", seq_score="Edited sequence"),
    "O-62_gravity_physics_data-generator": _s("Gravity motion", "Gravity motion is incorrect", process="Gravity process"),
    "O-64_animal_matching_data-generator": _s("Animal matching", "Animals were paired or moved incorrectly", core="Final animal placement"),
    "O-65_animal_size_sorting_data-generator": _s("Sort animals by size", "Animal size order or count is incorrect", arrangement="Size ordering"),
    "O-6_2d_geometric_transformation_data-generator": _s("2D geometric transform", "Final pose or orbital motion is incorrect", core="Final transformed pose"),
    "O-75_communicating_vessels_data-generator": _s("Communicating vessels", "Liquid equilibrium is incorrect", final_and_volume="Liquid result"),
    "O-85_2d_object_rotation_data-generator": _s("2D object rotation", "Object rotation is incorrect", task_score="Rotation quality"),
    "O-9_shape_scaling_data-generator": _s("Shape scaling", "Scaled result has the wrong size or shape", completion="Scaling completion"),
}


def semantics(task_name):
    return TASK_SEMANTICS[task_name]


def title_for(task_name, fallback):
    return TASK_SEMANTICS.get(task_name, {}).get("title", fallback)


def label_for(task_name, component_id, fallback):
    return TASK_SEMANTICS.get(task_name, {}).get("labels", {}).get(component_id, fallback)


def failure_for(task_name, component_id, fallback):
    item = TASK_SEMANTICS.get(task_name, {})
    return item.get("failures", {}).get(component_id, item.get("failure", fallback))


def component_failure_for(task_name, component_id, fallback):
    """Return an explicit secondary cause without repeating the main failure."""
    return TASK_SEMANTICS.get(task_name, {}).get("failures", {}).get(
        component_id, fallback)


def selection_target_for(task_name):
    return TASK_SEMANTICS.get(task_name, {}).get("labels", {}).get(
        "selection_target", "required target")


def selection_event_title(task_name, kind):
    labels = TASK_SEMANTICS.get(task_name, {}).get("labels", {})
    target = labels.get("selection_target", "required target")
    fallback = ("Incorrect %s selected" % target if kind == "wrong"
                else "%s not selected" % target.capitalize())
    return labels.get("selection_" + kind, fallback)
