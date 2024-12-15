# Given list of pose names
pose_names = """
Head-to-Knee_Forward_Bend_pose_or_Janu_Sirsasana_
Boat_Pose_or_Paripurna_Navasana_
Legs-Up-the-Wall_Pose_or_Viparita_Karani_
Reclining_Hand-to-Big-Toe_Pose_or_Supta_Padangusthasana_
Cockerel_Pose
Staff_Pose_or_Dandasana_
Supta_Baddha_Konasana_
Supta_Virasana_Vajrasana
Frog_Pose_or_Bhekasana
Chair_Pose_or_Utkatasana_
Warrior_I_Pose_or_Virabhadrasana_I_
Locust_Pose_or_Salabhasana_
Crane_(Crow)_Pose_or_Bakasana_
Bridge_Pose_or_Setu_Bandha_Sarvangasana_
Downward-Facing_Dog_pose_or_Adho_Mukha_Svanasana_
Revolved_Head-to-Knee_Pose_or_Parivrtta_Janu_Sirsasana_
Upward_Plank_Pose_or_Purvottanasana_
Shoulder-Pressing_Pose_or_Bhujapidasana_
Supported_Headstand_pose_or_Salamba_Sirsasana_
Handstand_pose_or_Adho_Mukha_Vrksasana_
Eight-Angle_Pose_or_Astavakrasana_
Feathered_Peacock_Pose_or_Pincha_Mayurasana_
Wild_Thing_pose_or_Camatkarasana_
Camel_Pose_or_Ustrasana_
Cobra_Pose_or_Bhujangasana_
Yogic_sleep_pose
Standing_Split_pose_or_Urdhva_Prasarita_Eka_Padasana_
Pose_Dedicated_to_the_Sage_Koundinya_or_Eka_Pada_Koundinyanasana_I_and_II
Wide-Angle_Seated_Forward_Bend_pose_or_Upavistha_Konasana_
Warrior_II_Pose_or_Virabhadrasana_II_
Sitting pose 1 (normal)
Plow_Pose_or_Halasana_
Extended_Revolved_Triangle_Pose_or_Utthita_Trikonasana_
Corpse_Pose_or_Savasana_
Upward_Bow_(Wheel)_Pose_or_Urdhva_Dhanurasana_
Cow_Face_Pose_or_Gomukhasana_
Dolphin_Plank_Pose_or_Makara_Adho_Mukha_Svanasana_
Scale_Pose_or_Tolasana_
Supported_Shoulderstand_pose_or_Salamba_Sarvangasana_
Standing_Forward_Bend_pose_or_Uttanasana_
Firefly_Pose_or_Tittibhasana_
Half_Lord_of_the_Fishes_Pose_or_Ardha_Matsyendrasana_
Standing_big_toe_hold_pose_or_Utthita_Padangusthasana
Plank_Pose_or_Kumbhakasana_
Fish_Pose_or_Matsyasana_
Akarna_Dhanurasana
Gate_Pose_or_Parighasana_
Tortoise_Pose
Bound_Angle_Pose_or_Baddha_Konasana_
Eagle_Pose_or_Garudasana_
Split pose
Bow_Pose_or_Dhanurasana_
Virasana_or_Vajrasana
Extended_Puppy_Pose_or_Uttana_Shishosana_
Side_Plank_Pose_or_Vasisthasana_
Side_Crane_(Crow)_Pose_or_Parsva_Bakasana_
Tree_Pose_or_Vrksasana_
Wide-Legged_Forward_Bend_pose_or_Prasarita_Padottanasana_
Scorpion_pose_or_vrischikasana
Half_Moon_Pose_or_Ardha_Chandrasana_
Low_Lunge_pose_or_Anjaneyasana_
Lord_of_the_Dance_Pose_or_Natarajasana_
Garland_Pose_or_Malasana_
Child_Pose_or_Balasana_
Dolphin_Pose_or_Ardha_Pincha_Mayurasana_
Bharadvaja's_Twist_pose_or_Bharadvajasana_I_
Four-Limbed_Staff_Pose_or_Chaturanga_Dandasana_
Warrior_III_Pose_or_Virabhadrasana_III_
Intense_Side_Stretch_Pose_or_Parsvottanasana_
Wind_Relieving_pose_or_Pawanmuktasana
Peacock_Pose_or_Mayurasana_
Noose_Pose_or_Pasasana_
viparita_virabhadrasana_or_reverse_warrior_pose
Happy_Baby_Pose_or_Ananda_Balasana_
Cat_Cow_Pose_or_Marjaryasana_
Side-Reclining_Leg_Lift_pose_or_Anantasana_
Pigeon_Pose_or_Kapotasana_
Heron_Pose_or_Krounchasana_
Upward_Facing_Two-Foot_Staff_Pose_or_Dwi_Pada_Viparita_Dandasana_
Extended_Revolved_Side_Angle_Pose_or_Utthita_Parsvakonasana_
Rajakapotasana
Seated_Forward_Bend_pose_or_Paschimottanasana_
"""

# Split and enumerate the pose names
pose_list = pose_names.strip().split("\n")
pose_dict = {i: pose_name for i, pose_name in enumerate(pose_list)}

# Save to an Excel file
import pandas as pd

df = pd.DataFrame(list(pose_dict.items()), columns=["Class ID", "Pose Name"])
df.to_excel("yoga_pose_classes.xlsx", index=False)

print("Pose names saved to yoga_pose_classes.xlsx")