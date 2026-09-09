import torch
from ultralytics import YOLO
from roboflow import Roboflow
import owlready2 as or2

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

onto = or2.get_ontology("mlprov.owx").load()

req_spec = onto["requirement_specification"]
train_data = onto["training_dataset"]
test_data = onto["testing_dataset"]
exp_model = onto["model"]
recall = onto["performance_metric"]
pred = onto["prediction"]

with onto:
    class description(or2.DataProperty):
        range = [str]

req1 = req_spec("req1")
req1.description = ["The model shall detect instances of the 'pie' class within images."]

model = YOLO("yolov8n.pt")

graph_model = exp_model("yolov8n.pt")

dataset = "/path/to/data/mutation_mutated.yolov8"   

orig_train_data = train_data("Original_Training_Data")
orig_train_data.description = ["https://github.com/AnonymousWriter1/BiasProvenance/label/mutation_unmutated.yolov8"]
mutated_train_data = train_data("Mutated_Training_Data")
mutated_train_data.description = ["https://github.com/AnonymousWriter1/BiasProvenance/label/mutation_mutated.yolov8"]
testing_data = test_data("Training_Validation_Set")                        
                
results = model.train(
    data=f"{dataset}/data.yaml",
    epochs=20,
    imgsz=640,
    batch=16,
    device=0,
    workers=0,
    verbose=True,
    augment=False,
    deterministic=True
)      

orig_pred_log = pred("Original_Prediction_Log")
orig_pred_log.description = ["https://github.com/AnonymousWriter1/BiasProvenance/label/m1v1"]
orig_recall = recall("Original_Recall")
orig_recall.description = ["1.00"]

mutated_pred_log = pred("Mutated_Prediction_Log")
mutated_pred_log.description = ["https://github.com/AnonymousWriter1/BiasProvenance/label/mutation_test-7"]
mutated_recall = recall("Mutated_Recall")
mutated_recall.description = ["0.73"]

onto.save(file="mlprov_mutation_instances.owl")