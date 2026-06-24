import torch
def predict_image(image,model,device,test_transform,class_names):
    image_tensor = test_transform(image.convert("RGB")).unsqueeze(0).to(device)
    with torch.no_grad():
        outputs = model(image_tensor)
        proba = torch.softmax(outputs,dim=1)
        top_proba , top_indices = torch.topk(proba,k=3)
        results = []
        for i in range(3):
            disease = class_names[top_indices[0][i].item()]
            confidence = (top_proba[0][i].item()*100)
            results.append((disease,confidence))
        return results

def format_name(name):
    return name.replace("_"," ")
