import cv2
from numpy import random
from collections import deque
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import math
import string
import torch
import torch.backends.cudnn as cudnn
import streamlit as st
import gc

from utils.google_utils import attempt_load
from utils.datasets import LoadStreams, LoadImages
from utils.general import (
    check_img_size, non_max_suppression, apply_classifier, scale_coords, xyxy2xywh, strip_optimizer)
from utils.plots import plot_one_box
from utils.torch_utils import select_device, load_classifier, time_synchronized

from models.models import *
from utils.datasets import *
from utils.general import *

from deep_sort_pytorch.utils.parser import get_config
from deep_sort_pytorch.deep_sort import DeepSort

import sqlite3
import smtplib
import ssl
import imghdr
from email.message import EmailMessage
import geocoder
from geopy.geocoders import Nominatim
import datetime


def load_classes(path):
    # Loads *.names file at 'path'
    with open(path, 'r') as f:
        names = f.read().split('\n')
    return list(filter(None, names))


global names
names = load_classes('data/coco.names')

# global weapons
# weapons = load_classes('data/custom.names')
# threat_weights = 'weights/custom.pt'

colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(len(names))]
palette = (2 ** 11 - 1, 2 ** 15 - 1, 2 ** 20 - 1)
data_deque = {}
speed_four_line_queue = {}
object_counter = {}

line2 = [(200, 500), (1050, 500)]


def xyxy_to_xywh(*xyxy):
    """" Calculates the relative bounding box from absolute pixel values. """
    bbox_left = min([xyxy[0].item(), xyxy[2].item()])
    bbox_top = min([xyxy[1].item(), xyxy[3].item()])
    bbox_w = abs(xyxy[0].item() - xyxy[2].item())
    bbox_h = abs(xyxy[1].item() - xyxy[3].item())
    x_c = (bbox_left + bbox_w / 2)
    y_c = (bbox_top + bbox_h / 2)
    w = bbox_w
    h = bbox_h
    return x_c, y_c, w, h


def xyxy_to_tlwh(bbox_xyxy):
    tlwh_bboxs = []
    for i, box in enumerate(bbox_xyxy):
        x1, y1, x2, y2 = [int(i) for i in box]
        top = x1
        left = y1
        w = int(x2 - x1)
        h = int(y2 - y1)
        tlwh_obj = [top, left, w, h]
        tlwh_bboxs.append(tlwh_obj)
    return tlwh_bboxs


def compute_color_for_labels(label):

    if label == 0:  # Person
        color = (85, 45, 255)
    elif label == 2:  # Car
        color = (222, 82, 175)
    elif label == 3:  # Motorbike
        color = (0, 204, 255)
    elif label == 5:  # Bus
        color = (0, 149, 255)
    else:
        color = [int((p * (label ** 2 - label + 1)) % 255) for p in palette]
    return tuple(color)


def draw_border(img, pt1, pt2, color, thickness, r, d):
    x1, y1 = pt1
    x2, y2 = pt2
    # Top left
    cv2.line(img, (x1 + r, y1), (x1 + r + d, y1), color, thickness)
    cv2.line(img, (x1, y1 + r), (x1, y1 + r + d), color, thickness)
    cv2.ellipse(img, (x1 + r, y1 + r), (r, r), 180, 0, 90, color, thickness)

    # Top right
    cv2.line(img, (x2 - r, y1), (x2 - r - d, y1), color, thickness)
    cv2.line(img, (x2, y1 + r), (x2, y1 + r + d), color, thickness)
    cv2.ellipse(img, (x2 - r, y1 + r), (r, r), 270, 0, 90, color, thickness)
    # Bottom left
    cv2.line(img, (x1 + r, y2), (x1 + r + d, y2), color, thickness)
    cv2.line(img, (x1, y2 - r), (x1, y2 - r - d), color, thickness)
    cv2.ellipse(img, (x1 + r, y2 - r), (r, r), 90, 0, 90, color, thickness)
    # Bottom right
    cv2.line(img, (x2 - r, y2), (x2 - r - d, y2), color, thickness)
    cv2.line(img, (x2, y2 - r), (x2, y2 - r - d), color, thickness)
    cv2.ellipse(img, (x2 - r, y2 - r), (r, r), 0, 0, 90, color, thickness)

    cv2.rectangle(img, (x1 + r, y1), (x2 - r, y2), color, -1, cv2.LINE_AA)
    cv2.rectangle(img, (x1, y1 + r), (x2, y2 - r - d), color, -1, cv2.LINE_AA)

    cv2.circle(img, (x1 + r, y1 + r), 2, color, 12)
    cv2.circle(img, (x2 - r, y1 + r), 2, color, 12)
    cv2.circle(img, (x1 + r, y2 - r), 2, color, 12)
    cv2.circle(img, (x2 - r, y2 - r), 2, color, 12)

    return img


def UI_box(x, img, color=None, label=None, line_thickness=None):
    # Plots one bounding box on image img
    tl = line_thickness or round(0.002 * (img.shape[0] + img.shape[1]) / 2) + 1
    color = color or [random.randint(0, 255) for _ in range(3)]
    c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))
    if label:
        tf = max(tl - 1, 1)
        t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
        img = draw_border(img, (c1[0], c1[1] - t_size[1] - 3), (c1[0] + t_size[0], c1[1] + 3), color, 1, 8, 2)
        cv2.putText(img, label, (c1[0], c1[1] - 2), 0, tl / 3, [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)


def estimateSpeed(location1, location2):
    d_pixels = math.sqrt(math.pow(location2[0] - location1[0], 2) + math.pow(location2[1] - location1[1], 2))
    ppm = 8
    d_meters = d_pixels / ppm
    time_constant = 15 * 3.6
    speed = d_meters * time_constant
    return speed


# Return true if line segments AB and CD intersect
def intersect(A, B, C, D):
    return ccw(A, C, D) != ccw(B, C, D) and ccw(A, B, C) != ccw(A, B, D)


def ccw(A, B, C):
    return (C[1] - A[1]) * (B[0] - A[0]) > (B[1] - A[1]) * (C[0] - A[0])


def draw_boxes(img, bbox, object_id, identities=None, offset=(0, 0)):

    height, width, _ = img.shape

    # Remove tracked point from buffer if object is lost
    for key in list(data_deque):
        if key not in identities:
            data_deque.pop(key)

    for i, box in enumerate(bbox):
        x1, y1, x2, y2 = [int(i) for i in box]
        x1 += offset[0]
        x2 += offset[0]
        y1 += offset[1]
        y2 += offset[1]

        box_area = (x2-x1) * (y2-y1)
        box_height = (y2 - y1)

        center = (int((x2 + x1) / 2), int((y2 + y2) / 2))

        # get ID of object
        id = int(identities[i]) if identities is not None else 0

        # create new buffer for new object
        if id not in data_deque:
            data_deque[id] = deque(maxlen=64)
            speed_four_line_queue[id] = []

        color = compute_color_for_labels(object_id[i])
        obj_name = names[object_id[i]]
        label = str(id) + ' %s' % (obj_name)

        # Add center to buffer
        data_deque[id].appendleft(center)

        # print("id ", id)
        # print("data_deque[id] ", data_deque[id])

        if len(data_deque[id]) >= 2:
            # print("data_deque[id][i-1]", data_deque[id][1], data_deque[id][0])

            if intersect(data_deque[id][0], data_deque[id][1], line2[0], line2[1]):

                obj_speed = estimateSpeed(data_deque[id][1], data_deque[id][0])

                speed_four_line_queue[id].append(obj_speed)

                if obj_name not in object_counter:
                    object_counter[obj_name] = 1
                else:
                    object_counter[obj_name] += 1

        try:
            # label = label + " " + str(sum(speed_four_line_queue[id])//len(speed_four_line_queue[id]))
            label = label
        except:
            pass

        UI_box(box, img, label=label, color=color, line_thickness=2)

        # draw trail
        for i in range(1, len(data_deque[id])):
            # check if on buffer value is none
            if data_deque[id][i - 1] is None or data_deque[id][i] is None:
                continue

            # Generate dynamic thickness of trails
            thickness = int(np.sqrt(64 / float(i + i)) * 1.5)

            # draw trails
            cv2.line(img, data_deque[id][i - 1], data_deque[id][i], color, thickness)

    count = 0
    for idx, (key, value) in enumerate(object_counter.items()):
        # print(idx, key, value)
        cnt_str = str(key) + ": " + str(value)

        count += value

    return img, count, id


def analytics(data):

    st.table(data)
    graph1, graph2 = st.columns(2)
    data.sort_values(by=['Number'], inplace=True)

    # Pie Chart
    fig1, ax1 = plt.subplots()
    ax1.pie(data['Number'].tolist(), labels=data['Objects'].tolist(), autopct='%1.1f%%', shadow=True, startangle=90)
    ax1.axis('equal')
    graph1.pyplot(fig1)

    # Bar Chart
    fig2 = plt.figure()
    ax2 = fig2.add_axes([0, 0, 1, 1])
    ax2.set_ylabel('Number')
    ax2.set_xlabel('Objects')
    ax2.bar(data['Objects'].tolist(), data['Number'].tolist())
    graph2.pyplot(fig2)


def send_report(threat_type, evidence):

    try:

        Sender_Email = "aditya24102001@gmail.com"
        Reciever_Email = "threatdetectoremail@gmail.com"
        Password = "wykmshsevqxzmxvq"  # App password
        # https://support.google.com/mail/answer/185833?hl=en-GB#:~:text=Create%20and%20use%20app%20passwords&text=Go%20to%20your%20Google%20Account

        # Get date/time and location
        datetime_obj = datetime.datetime.now()
        loc = Nominatim(user_agent="GetLoc")
        getLoc = loc.geocode("Nottingham")
        address = getLoc.address

        message = "Detected possession of " + threat_type \
                  + ".\nLocation: " + address \
                  + "\nDate/Time: " + str(datetime_obj)

        newMessage = EmailMessage()
        newMessage['Subject'] = "Threat Detected"
        newMessage['From'] = Sender_Email
        newMessage['To'] = Reciever_Email
        newMessage.set_content(message)

        with open(evidence, 'rb') as f:
            image_data = f.read()
            image_type = imghdr.what(f.name)
            image_name = f.name

        newMessage.add_attachment(image_data, maintype='image', subtype=image_type, filename=image_name)

        with smtplib.SMTP_SSL('smtp.gmail.com', 465) as smtp:
            smtp.login(Sender_Email, Password)
            smtp.send_message(newMessage)
        print("Threat sent to authorities")

    except:
        print("[Warning!] Error sending email.")


def add_threatData(conn, cursor, threat_info):

    id, obj = threat_info

    g = geocoder.ip('me')
    coordinates = g.latlng
    lat = coordinates[0]
    lon = coordinates[1]
    ll = "%s,%s" % (str(lat), str(lon))
    geoLoc = Nominatim(user_agent="GetLoc")
    loc = geoLoc.reverse(ll)
    location = loc.address

    dt = datetime.datetime.now()
    date = dt.strftime("%x")
    time = dt.strftime("%X")

    cursor.executemany("insert into threats values (?,?,?,?,?)", [(id, obj, location, str(date), str(time))])
    conn.commit()
    print("Threat inserted into database")


def load_yolor_video(vid_name, enable_GPU, confidence, assigned_class_id, kpi, stframe, database):

    conn, cursor = database
    data_deque.clear()
    speed_four_line_queue.clear()
    object_counter.clear()

    kpi1_text, kpi2_text, kpi3_text = kpi

    out, source, weights, save_txt, imgsz, cfg = 'inference/output', vid_name, 'yolor_p6.pt', False, 1280, 'cfg/yolor_p6.cfg'

    # Initialize DeepSORT
    cfg_deep = get_config()
    cfg_deep.merge_from_file("deep_sort_pytorch/configs/deep_sort.yaml")
    deepsort = DeepSort(cfg_deep.DEEPSORT.REID_CKPT,
                        max_dist=cfg_deep.DEEPSORT.MAX_DIST, min_confidence=cfg_deep.DEEPSORT.MIN_CONFIDENCE,
                        nms_max_overlap=cfg_deep.DEEPSORT.NMS_MAX_OVERLAP,
                        max_iou_distance=cfg_deep.DEEPSORT.MAX_IOU_DISTANCE,
                        max_age=cfg_deep.DEEPSORT.MAX_AGE, n_init=cfg_deep.DEEPSORT.N_INIT,
                        nn_budget=cfg_deep.DEEPSORT.NN_BUDGET,
                        use_cuda=True)

    # Initialize GPU
    if enable_GPU:
        torch.cuda.memory_summary(device=None, abbreviated=False)
        gc.collect()
        torch.cuda.empty_cache()

        device = select_device('gpu')
    else:
        device = select_device('cpu')

    if os.path.exists(out):
        shutil.rmtree(out)
    os.makedirs(out)
    half = device.type != 'cpu'

    # Load model
    model = Darknet(cfg, imgsz)
    model.load_state_dict(torch.load(weights, map_location=device)['model'])
    model.to(device).eval()
    if half:
        model.half()

    # Second-stage classifier
    classify = False
    if classify:
        modelc = load_classifier(name='resnet101', n=2)
        modelc.load_state_dict(torch.load('weights/resnet101.pt', map_location=device)['model'])
        modelc.to(device).eval()

    # Set Dataloader
    vid_path, vid_writer = None, None

    save_img = True
    dataset = LoadImages(source, img_size=imgsz, auto_size=64)

    # Run inference
    t0 = time.time()
    img = torch.zeros((1, 3, imgsz, imgsz), device=device)
    _ = model(img.half() if half else img) if device.type != 'cpu' else None
    prevTime = 0
    count = 0

    # Extract threat data
    status = None
    class_count = dict()
    class_id = dict()
    threats = ["handgun", "long gun", "knife"]

    for path, img, im0s, vid_cap in dataset:

        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()
        img /= 255.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Inference
        t1 = time_synchronized()
        print(img.shape)

        pred = model(img)[0]

        # Apply NMS
        pred = non_max_suppression(pred, confidence, 0.5, classes=assigned_class_id, agnostic=False)
        t2 = time_synchronized()

        # Apply Classifier
        if classify:
            pred = apply_classifier(pred, modelc, img, im0s)

        detected_threats = []
        obj = None

        # Process detections
        for i, det in enumerate(pred):
            p, s, im0 = path, '', im0s

            save_path = str(Path(out) / Path(p).name)
            txt_path = str(Path(out) / Path(p).stem) + ('_%g' % dataset.frame if dataset.mode == 'video' else '')
            s += '%gx%g ' % img.shape[2:]
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]
            if det is not None and len(det):
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()
                    s += '%g %ss, ' % (n, names[int(c)])
                    class_count[names[int(c)]] = '%g' % n

                    # Alert if weapon possession is detected
                    if names[int(c)] in threats:
                        obj = names[int(c)]
                        detected_threats.append(obj)

                    for threat in threats:
                        isSafe = False if threat in class_count.keys() else True

                        if isSafe:
                            status = "Safe"
                            kpi3_text.empty()
                            kpi3_text.write(f"<h1 style = 'text-align: center; color: green;'>{status}</h1>",
                                            unsafe_allow_html=True)
                        else:
                            status = "Danger"
                            kpi3_text.empty()
                            kpi3_text.write(f"<h1 style = 'text-align: center; color: red;'>{status}</h1>",
                                            unsafe_allow_html=True)
                            break

                xywh_bboxs = []
                confs = []
                oids = []
                for *xyxy, conf, cls in det:
                    x_c, y_c, bbox_w, bbox_h = xyxy_to_xywh(*xyxy)
                    xywh_obj = [x_c, y_c, bbox_w, bbox_h]
                    xywh_bboxs.append(xywh_obj)
                    confs.append([conf.item()])
                    oids.append(int(cls))

                    if save_txt:
                        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()
                        with open(txt_path + '.txt', 'a') as f:
                            f.write(('%g ' * 5 + '\n') % (cls, *xywh))

                xywhs = torch.Tensor(xywh_bboxs)
                confss = torch.Tensor(confs)

                outputs = deepsort.update(xywhs, confss, oids, im0)
                if len(outputs) > 0:
                    bbox_xyxy = outputs[:, :4]
                    identities = outputs[:, -2]
                    object_id = outputs[:, -1]
                    im0, count, id = draw_boxes(im0, bbox_xyxy, object_id, identities)

                    # Validate object to identify as threat
                    if status is "Danger" and id not in class_id.keys():
                        for threat in detected_threats:
                            # Caputure image containing threat
                            evidence = 'inference/threats/threat.jpg'
                            cv2.imwrite(evidence, im0)

                            # Store threat in database
                            threat_info = (id, threat)
                            add_threatData(conn, cursor, threat_info)

                            # Report threat to local authorities
                            message = "%s [ID %s]" % (threat, str(id))
                            send_report(message, evidence)
                            class_id[id] = threat

            print('%sDone. (%.3fs)' % (s, t2 - t1))

            currTime = time.time()
            fps = 1 / (currTime - prevTime)
            prevTime = currTime
            cv2.line(im0, (20, 25), (127, 25), [85, 45, 255], 30)
            cv2.putText(im0, f'FPS: {int(fps)}', (11, 35), 0, 1, [225, 255, 255], thickness=2, lineType=cv2.LINE_AA)

            # Save results
            if save_img:
                if dataset.mode == 'images':
                    cv2.imwrite(save_path, im0)
                else:
                    if vid_path != save_path:
                        vid_path = save_path
                        if isinstance(vid_writer, cv2.VideoWriter):
                            vid_writer.release()

                        fourcc = 'mp4v'
                        fps = vid_cap.get(cv2.CAP_PROP_FPS)
                        w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                        h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*fourcc), fps, (w, h))
                    vid_writer.write(im0)

        kpi1_text.write(f"<h1 style = 'text-align: center; color: green;'>{'{:.1}'.format(fps)}</h1>", unsafe_allow_html=True)
        kpi2_text.write(f"<h1 style = 'text-align: center; color: green;'>{len(data_deque)}</h1>", unsafe_allow_html=True)

        stframe.image(im0, channels='BGR', use_column_width=True)

        data = pd.DataFrame({
            'Objects': class_count.keys(),
            'Number': class_count.values()
        })
        analytics(data)

    if save_txt or save_img:
        print('Results saved to %s' % Path(out))

    print('Done. (%.3fs)' % (time.time() - t0))
    cv2.destroyAllWindows()


def load_yolor_image(path, assigned_class_id):

    source, cfg, weights, conf, img_size, device = path, 'cfg/yolor_p6.cfg', 'yolor_p6.pt', 0.25, 1280, 'gpu'

    device = select_device(device)
    half = device.type != 'cpu'

    model = Darknet(cfg, img_size)
    model.load_state_dict(torch.load(weights, map_location=device)['model'])
    model.to(device).eval()
    if half:
        model.half()

    classify = False
    if classify:
        modelc = load_classifier(name='resnet101', n=2)
        modelc.load_state_dict(torch.load('weights/resnet101.pt', map_location=device)['model'])

    save_img = True
    dataset = LoadImages(source, img_size=img_size, auto_size=64)

    colours = [[random.randint(0, 255) for _ in range(3)] for _ in range(len(names))]

    t0 = time.time()
    img = torch.zeros((1, 3, img_size, img_size), device=device)
    _ = model(img.half() if half else img) if device.type != 'cpu' else None
    for path, img, im0s, vid_cap in dataset:
        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()
        img /= 255.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Inference
        t1 = time_synchronized()
        pred = model(img)[0]

        # Apply NMS
        pred = non_max_suppression(pred, conf, 0.5, classes=assigned_class_id, agnostic=False)
        t2 = time_synchronized()

        # Apply Classifier
        if classify:
            pred = apply_classifier(pred, modelc, img, im0s)

        # Process detections
        for i, det in enumerate(pred):
            p, s, im0 = path, '', im0s

            save_path = str(Path('inference/output') / Path(p).name)
            txt_path = str(Path('inference/output') / Path(p).stem) + ('_%g' % dataset.frame if dataset.mode == 'video' else '')
            s += '%gx%g ' % img.shape[2:]
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]
            class_count = dict()
            if det is not None and len(det):
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()
                    s += '%g %ss, ' % (n, names[int(c)])
                    class_count[names[int(c)]] = '%g' % n

                # Write results
                for *xyxy, conf, cls in det:
                    if save_img:
                        label = '%s %.2f' % (names[int(cls)], conf)
                        plot_one_box(xyxy, im0, label=label, color=colors[int(cls)], line_thickness=3)

            if save_img:
                if dataset.mode == 'images':
                    cv2.imwrite(save_path, im0)

            st_frame = st.empty()
            st_frame.image(im0, channels='BGR', use_column_width=True)

            if class_count:
                data = pd.DataFrame({
                    'Objects': class_count.keys(),
                    'Number': class_count.values()
                })
                analytics(data)


def load_yolor_camera(enable_GPU, confidence, assigned_class_id, kpi, stframe, database, stop=None):

    conn, cursor = database
    data_deque.clear()
    speed_four_line_queue.clear()
    object_counter.clear()

    kpi1_text, kpi2_text, kpi3_text = kpi

    out, source, weights, save_txt, imgsz, cfg = 'inference/output', '0', 'yolor_p6.pt', False, 1280, 'cfg/yolor_p6.cfg'

    # Initialize DeepSORT
    cfg_deep = get_config()
    cfg_deep.merge_from_file("deep_sort_pytorch/configs/deep_sort.yaml")
    deepsort = DeepSort(cfg_deep.DEEPSORT.REID_CKPT,
                        max_dist=cfg_deep.DEEPSORT.MAX_DIST, min_confidence=cfg_deep.DEEPSORT.MIN_CONFIDENCE,
                        nms_max_overlap=cfg_deep.DEEPSORT.NMS_MAX_OVERLAP,
                        max_iou_distance=cfg_deep.DEEPSORT.MAX_IOU_DISTANCE,
                        max_age=cfg_deep.DEEPSORT.MAX_AGE, n_init=cfg_deep.DEEPSORT.N_INIT,
                        nn_budget=cfg_deep.DEEPSORT.NN_BUDGET,
                        use_cuda=True)

    # Initialize GPU
    if enable_GPU:
        torch.cuda.memory_summary(device=None, abbreviated=False)
        gc.collect()
        torch.cuda.empty_cache()

        device = select_device('gpu')
    else:
        device = select_device('cpu')

    if os.path.exists(out):
        shutil.rmtree(out)
    os.makedirs(out)
    half = device.type != 'cpu'

    # Load model
    model = Darknet(cfg, imgsz)
    model.load_state_dict(torch.load(weights, map_location=device)['model'])
    model.to(device).eval()
    if half:
        model.half()

    # Second-stage classifier
    classify = False
    if classify:
        modelc = load_classifier(name='resnet101', n=2)
        modelc.load_state_dict(torch.load('weights/resnet101.pt', map_location=device)['model'])
        modelc.to(device).eval()

    # Set Dataloader
    vid_path, vid_writer = None, None

    save_img = True
    cudnn.benchmark = True
    dataset = LoadStreams(source, img_size=imgsz)

    # Run inference
    t0 = time.time()
    img = torch.zeros((1, 3, imgsz, imgsz), device=device)
    _ = model(img.half() if half else img) if device.type != 'cpu' else None
    prevTime = 0
    count = 0

    # Extract threat data
    status = None
    class_count = dict()
    class_id = dict()
    threats = ["handgun", "long gun", "knife"]

    for path, img, im0s, vid_cap in dataset:

        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()
        img /= 255.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Inference
        t1 = time_synchronized()
        print(img.shape)

        pred = model(img)[0]

        # Apply NMS
        pred = non_max_suppression(pred, confidence, 0.5, classes=assigned_class_id, agnostic=False)
        t2 = time_synchronized()

        # Apply Classifier
        if classify:
            pred = apply_classifier(pred, modelc, img, im0s)

        detected_threats = []
        obj = None

        # Process detections
        for i, det in enumerate(pred):
            p, s, im0 = path[i], '%g: ' % i, im0s[i].copy()

            save_path = str(Path(out) / Path(p).name)
            txt_path = str(Path(out) / Path(p).stem) + ('_%g' % dataset.frame if dataset.mode == 'video' else '')
            s += '%gx%g ' % img.shape[2:]
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]
            if det is not None and len(det):
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()
                    s += '%g %ss, ' % (n, names[int(c)])
                    class_count[names[int(c)]] = '%g' % n
                    print(names[int(c)])

                    # Alert if weapon possession is detected
                    threats = ["handgun", "long gun", "knife"]
                    for threat in threats:
                        isSafe = False if threat in class_count.keys() else True

                        if isSafe:
                            status = "Safe"
                            kpi3_text.empty()
                            kpi3_text.write(f"<h1 style = 'text-align: center; color: green;'>{status}</h1>",
                                            unsafe_allow_html=True)
                        else:
                            status = "Danger"
                            kpi3_text.empty()
                            kpi3_text.write(f"<h1 style = 'text-align: center; color: red;'>{status}</h1>",
                                            unsafe_allow_html=True)
                            break

                xywh_bboxs = []
                confs = []
                oids = []
                for *xyxy, conf, cls in det:
                    x_c, y_c, bbox_w, bbox_h = xyxy_to_xywh(*xyxy)
                    xywh_obj = [x_c, y_c, bbox_w, bbox_h]
                    xywh_bboxs.append(xywh_obj)
                    confs.append([conf.item()])
                    oids.append(int(cls))

                    if save_txt:
                        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()
                        with open(txt_path + '.txt', 'a') as f:
                            f.write(('%g ' * 5 + '\n') % (cls, *xywh))

                xywhs = torch.Tensor(xywh_bboxs)
                confss = torch.Tensor(confs)

                outputs = deepsort.update(xywhs, confss, oids, im0)
                if len(outputs) > 0:
                    bbox_xyxy = outputs[:, :4]
                    identities = outputs[:, -2]
                    object_id = outputs[:, -1]
                    im0, count, id = draw_boxes(im0, bbox_xyxy, object_id, identities)

                    # Validate object to identify as threat
                    if status is "Danger" and id not in class_id.keys():
                        for threat in detected_threats:
                            # Caputure image containing threat
                            evidence = 'inference/threats/threat.jpg'
                            cv2.imwrite(evidence, im0)

                            # Store threat in database
                            threat_info = (id, threat)
                            add_threatData(conn, cursor, threat_info)

                            # Report threat to local authorities
                            message = "%s [ID %s]" % (threat, str(id))
                            send_report(message, evidence)
                            class_id[id] = threat

            print('%sDone. (%.3fs)' % (s, t2 - t1))

            currTime = time.time()
            fps = 1 / (currTime - prevTime)
            prevTime = currTime
            cv2.line(im0, (20, 25), (127, 25), [85, 45, 255], 30)
            cv2.putText(im0, f'FPS: {int(fps)}', (11, 35), 0, 1, [225, 255, 255], thickness=2, lineType=cv2.LINE_AA)

            # Save results
            # if save_img:
            #     if dataset.mode == 'images':
            #         cv2.imwrite(save_path, im0)
            #     else:
            #         if vid_path != save_path:
            #             vid_path = save_path
            #             if isinstance(vid_writer, cv2.VideoWriter):
            #                 vid_writer.release()
            #
            #             fourcc = 'mp4v'
            #             fps = vid_cap.get(cv2.CAP_PROP_FPS)
            #             w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            #             h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            #             vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*fourcc), fps, (w, h))
            #         vid_writer.write(im0)

        kpi1_text.write(f"<h1 style = 'text-align: center; color: green;'>{'{:.1}'.format(fps)}</h1>", unsafe_allow_html=True)
        kpi2_text.write(f"<h1 style = 'text-align: center; color: green;'>{len(data_deque)}</h1>", unsafe_allow_html=True)

        stframe.image(im0, channels='BGR', use_column_width=True)

        data = pd.DataFrame({
            'Objects': class_count.keys(),
            'Number': class_count.values()
        })
        analytics(data)

        if stop:
            del vid_cap
            cv2.destroyAllWindows()
            break
