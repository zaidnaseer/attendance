from deepface import DeepFace
import os
import json 

name = input("Enter name of student: ")

# if not os.path.exists(
#         f"images/{name}"
#     ):
#     print("Directory does not exist!")
#     exit(1)

# print(
#     DeepFace.verify(img1_path="images/zaid/pic1.jpg", img2_path="images/zaid/anas.png", model_name="Dlib")
# )

print(
    DeepFace.stream(
        db_path=f"images/zaid",
        enable_face_analysis = False,
        model_name="Dlib",
        source=0,
        anti_spoofing=True
    )
)

