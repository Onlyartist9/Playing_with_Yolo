# import libraries
import streamlit as st
from transformers import YolosImageProcessor, YolosForObjectDetection
from PIL import Image,ImageDraw,ImageFont
import torch

# load model and image processor
model = YolosForObjectDetection.from_pretrained('hustvl/yolos-tiny')
image_processor = YolosImageProcessor.from_pretrained('hustvl/yolos-tiny')

# create a file uploader widget
uploaded_file = st.file_uploader("Choose an image file", type=["jpg", "png"])

# if a file is uploaded, process it and display the results
if uploaded_file is not None:
    # load the image from the file
    image = Image.open(uploaded_file)
    # display the original image
    st.image(image, caption="Original Image", use_column_width=True)
    # process the image with the model
    inputs = image_processor(images=image, return_tensors="pt")
    outputs = model(**inputs)
    # get the predicted bounding boxes and classes
    logits = outputs.logits
    bboxes = outputs.pred_boxes
    # post-process the results and filter by confidence threshold
    target_sizes = torch.tensor([image.size[::-1]])
    results = image_processor.post_process_object_detection(outputs, threshold=0.9, target_sizes=target_sizes)[0]
    # draw the bounding boxes on the image
    annotated_image = image.copy()
    draw = ImageDraw.Draw(annotated_image)
    for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
        # convert the box coordinates to integers
        box = [int(i) for i in box.tolist()]
        # draw a rectangle around the object
        draw.rectangle(box, outline="red", width=3)
        # write the class name and confidence score above the object
        text = f"{model.config.id2label[label.item()]}: {round(score.item(), 3)}"
        font = ImageFont.truetype("/usr/share/fonts/truetype/freefont/FreeMono.ttf", size=16)
        text_width, text_height = font.getsize(text)
        draw.rectangle([box[0], box[1] - text_height - 4, box[0] + text_width + 4, box[1]], fill="red")
        draw.text((box[0] + 2, box[1] - text_height - 2), text, fill="white", font=font)
    # display the annotated image
    st.image(annotated_image, caption="Annotated Image", use_column_width=True)
