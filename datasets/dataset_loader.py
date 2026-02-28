from roboflow import Roboflow

rf = Roboflow(api_key="RLYarGTwCcbFdXc30EPd")
project = rf.workspace("ankit-shrivastava-d2cx1").project("hazy-vqiq1")
version = project.version(2)
dataset = version.download("yolov8")
