POSE_FEEDBACK = {
    "tree": "Make sure your spine is straight and arms are raised.",
    "warrior": "Ensure your front knee is bent at a right angle and arms are aligned.",
    "downward_dog": "Keep your heels touching the floor and spine extended.",
    "child_pose": "Relax your back and ensure your arms are stretched forward."
}

def get_pose_feedback(prediction, pose=None):
    """Generate feedback based on pose prediction."""
    if not pose or pose not in POSE_FEEDBACK:
        return "No specific feedback available for the selected pose."

    # Customize feedback with prediction-based suggestions if needed
    feedback = POSE_FEEDBACK[pose]
    return feedback
