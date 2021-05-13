import logging as log
import os.path as osp
import sys
import time
from argparse import ArgumentParser
import cv2
import numpy as np
import shutil
import paho.mqtt.publish as publish
import time
import json
import os
from openvino.inference_engine import IENetwork, IECore
from src.ie_module import InferenceContext
from src.object_detector import ObjectDetector
from src.person_attributes import PersonAttributes
from src.semantic_segmentation import SemanticSegmentation
from src.instance_segmentation import InstanceSegmentation
import json
import base64
import random
import threading ,queue
from _thread import start_new_thread

DEVICE_KINDS = ['CPU', 'GPU', 'FPGA', 'MYRIAD', 'HETERO', 'HDDL']

def build_argparser():
    parser = ArgumentParser()

    general = parser.add_argument_group('General')
    general.add_argument('-i', '--input', metavar="PATH", default='0',
                         help="(optional) Path to the input video " \
                         "('0' for the camera, default)")
    general.add_argument('-o', '--output', metavar="PATH", default="",
                         help="(optional) Path to save the output video to")
    general.add_argument('--no_show', action='store_true',
                         help="(optional) Do not display output")
    general.add_argument('--segmentation_weight', default=0.5, type=float,
                         help="(optional) Weight to be given to the segmentation mask." \
                         "Higher weight will decrease the opacity.")
    general.add_argument('-tl', '--timelapse', action='store_true',
                         help="(optional) Auto-pause after each frame")
    general.add_argument('-cw', '--crop_width', default=0, type=int,
                         help="(optional) Crop the input stream to this width " \
                         "(default: no crop). Both -cw and -ch parameters " \
                         "should be specified to use crop.")
    general.add_argument('-ch', '--crop_height', default=0, type=int,
                         help="(optional) Crop the input stream to this height " \
                         "(default: no crop). Both -cw and -ch parameters " \
                         "should be specified to use crop.")

    models = parser.add_argument_group('Models')
    models.add_argument('-m_pv', metavar="PATH", default="models/pedestrian-and-vehicle-detector-adas-0001.xml",
                        help="Path to the person and vehicle detection XML file")
    models.add_argument('-m_pa', metavar="PATH", default="models/person-attributes-recognition-crossroad-0230.xml",
                        help="Path to the Person Attributes detection model XML file")
    models.add_argument('-m_rs', metavar="PATH", default="models/semantic-segmentation-adas-0001.xml",
                        help="Path to the Roadside object segmentation model XML file")
    models.add_argument('-m_is', metavar="PATH", default="models/instance-segmentation-security-0050.xml",
                        help="Path to the Instance (COCO Dataset) Segmentation model XML file")
                        
    models.add_argument('-fd_iw', '--fd_input_width', default=0, type=int,
                         help="(optional) specify the input width of detection model " \
                         "(default: use default input width of model). Both -fd_iw and -fd_ih parameters " \
                         "should be specified for reshape.")
    models.add_argument('-fd_ih', '--fd_input_height', default=0, type=int,
                         help="(optional) specify the input height of detection model " \
                         "(default: use default input height of model). Both -fd_iw and -fd_ih parameters " \
                         "should be specified for reshape.")
    models.add_argument('-is_labels', default="coco_labels.txt", metavar="PATH",
                         help="(optional) path to the labels used for Instance Segmentation Model")
    
    infer = parser.add_argument_group('Inference options')
    infer.add_argument('-d_pv', default='CPU', choices=DEVICE_KINDS,
                       help="(optional) Target device for the " \
                       "Face Detection model (default: %(default)s)")
    infer.add_argument('-d_pa', default='CPU', choices=DEVICE_KINDS,
                       help="(optional) Target device for the " \
                       "Person Attributes Detection model (default: %(default)s)")
    infer.add_argument('-d_rs', default='CPU', choices=DEVICE_KINDS,
                       help="(optional) Target device for the " \
                       "Roadside Object Segmentation model (default: %(default)s)")
    infer.add_argument('-d_is', default='CPU', choices=DEVICE_KINDS,
                       help="(optional) Target device for the " \
                       "Instance (COCO Dataset) Segmentation model (default: %(default)s)")
    infer.add_argument('-l', '--cpu_lib', metavar="PATH", default="",
                       help="(optional) For MKLDNN (CPU)-targeted custom layers, if any. " \
                       "Path to a shared library with custom layers implementations")
    infer.add_argument('-c', '--gpu_lib', metavar="PATH", default="",
                       help="(optional) For clDNN (GPU)-targeted custom layers, if any. " \
                       "Path to the XML file with descriptions of the kernels")
    infer.add_argument('-v', '--verbose', action='store_true',
                       help="(optional) Be more verbose")
    infer.add_argument('-pc', '--perf_stats', action='store_true',
                       help="(optional) Output detailed per-layer performance stats")
    infer.add_argument('-t_pv', metavar='[0..1]', type=float, default=0.3,
                       help="(optional) Probability threshold for person & vehicle detections" \
                       "(default: %(default)s)")
    infer.add_argument('-exp_r_pv', metavar='NUMBER', type=float, default=1.15,
                       help="(optional) Scaling ratio for bboxes passed to face recognition " \
                       "(default: %(default)s)")
    return parser


class FrameProcessor:
    QUEUE_SIZE = 16

    def __init__(self, args):
        used_devices = set([args.d_pv, args.d_pa, args.d_rs])
        self.context = InferenceContext()
        context = self.context
        context.load_plugins(used_devices, args.cpu_lib, args.gpu_lib)
        for d in used_devices:
            context.get_plugin(d).set_config({
                "PERF_COUNT": "YES" if args.perf_stats else "NO"})

        log.info("Loading models")

        pv_detector_net = self.load_model(args.m_pv)
        attribute_detector_net = self.load_model(args.m_pa)
        # road_segmentation_net = self.load_model(args.m_rs)
        instance_segmentation_net = self.load_model(args.m_is)
        
        assert (args.fd_input_height and args.fd_input_width) or \
               (args.fd_input_height==0 and args.fd_input_width==0), \
            "Both -fd_iw and -fd_ih parameters should be specified for reshape"
        
        if args.fd_input_height and args.fd_input_width :
            pv_detector_net.reshape({"data": [1, 3, args.fd_input_height,args.fd_input_width]})

        pv_labels = ['Person', 'Vehicle', 'check_labels']
        self.pv_detector = ObjectDetector(pv_detector_net, pv_labels, confidence_threshold=args.t_pv,
                                          roi_scale_factor=args.exp_r_pv)
        self.attribute_detector = PersonAttributes(attribute_detector_net)
        # self.road_segmentation = SemanticSegmentation(road_segmentation_net)
        self.instance_segmentation = InstanceSegmentation(instance_segmentation_net, args.is_labels)

        self.pv_detector.deploy(args.d_pv, context, queue_size=self.QUEUE_SIZE)
        self.attribute_detector.deploy(args.d_pa, context, queue_size=self.QUEUE_SIZE)
        # self.road_segmentation.deploy(args.d_rs, context, queue_size=self.QUEUE_SIZE)
        self.instance_segmentation.deploy(args.d_is, context, queue_size=self.QUEUE_SIZE)
        log.info("Models are loaded")

    def load_model(self, model_path):
        model_path = osp.abspath(model_path)
        model_weights_path = osp.splitext(model_path)[0] + ".bin"
        log.info("Loading the model from '%s'" % (model_path))
        assert osp.isfile(model_path), \
            "Model description is not found at '%s'" % (model_path)
        assert osp.isfile(model_weights_path), \
            "Model weights are not found at '%s'" % (model_weights_path)
        model = IENetwork(model_path, model_weights_path)
        # ie = IECore()
        # model = ie.read_network(model_path, model_weights_path)
        log.info("Model is loaded")
        return model

    def process(self, frame):
        assert len(frame.shape) == 3, \
            "Expected input frame in (H, W, C) format"
        assert frame.shape[2] in [3, 4], \
            "Expected BGR or BGRA input"

        orig_image = frame.copy()
        frame = frame.transpose((2, 0, 1)) # HWC to CHW
        frame = np.expand_dims(frame, axis=0)

        self.pv_detector.clear()
        self.attribute_detector.clear()
        # self.road_segmentation.clear()
        self.instance_segmentation.clear()

        self.pv_detector.start_async(frame)
        rois = self.pv_detector.get_roi_proposals(frame)
        if self.QUEUE_SIZE < len(rois):
            log.warning("Too many faces for processing." \
                    " Will be processed only %s of %s." % \
                    (self.QUEUE_SIZE, len(rois)))
            rois = rois[:self.QUEUE_SIZE]
        
        self.attribute_detector.start_async(frame, rois)
        attributes = self.attribute_detector.get_attributes()

        # self.road_segmentation.start_async(frame)
        # segmentation_mask = self.road_segmentation.get_class_map()

        self.instance_segmentation.start_async(frame)
        object_masks = self.instance_segmentation.object_masks()

        # outputs = [[rois, attributes], segmentation_mask, object_masks]
        outputs = [[rois, attributes], None, object_masks]
        return outputs


    def get_performance_stats(self):
        stats = {
            'pv_detector': self.pv_detector.get_performance_stats(),
        }
        return stats


class Visualizer:
    BREAK_KEY_LABELS = "q(Q) or Escape"
    BREAK_KEYS = {ord('q'), ord('Q'), 27}


    def __init__(self, args):
        self.frame_processor = FrameProcessor(args)
        self.display = not args.no_show
        self.print_perf_stats = args.perf_stats
        self.segmentation_weight = args.segmentation_weight

        self.frame_time = 0
        self.frame_start_time = time.perf_counter()
        self.fps = 0
        self.frame_num = 0
        self.frame_count = -1
        global device_id
        global upload_video
        person=0
        car=0
        animal=0

        self.input_crop = None
        if args.crop_width and args.crop_height:
            self.input_crop = np.array((args.crop_width, args.crop_height))

        self.frame_timeout = 0 if args.timelapse else 1
        
    def update_fps(self):
        now = time.perf_counter()
        self.frame_time = max(now - self.frame_start_time, sys.float_info.epsilon)
        self.fps = 1.0 / self.frame_time
        self.frame_start_time = now

    def draw_text_with_background(self, frame, text, origin,
                                  font=cv2.FONT_HERSHEY_SIMPLEX, scale=1.0,
                                  color=(0, 0, 0), thickness=1, bgcolor=(255, 255, 255)):
        text_size, baseline = cv2.getTextSize(text, font, scale, thickness)
        cv2.rectangle(frame,
                      tuple((origin + (0, baseline)).astype(int)),
                      tuple((origin + (text_size[0], -text_size[1])).astype(int)),
                      bgcolor, cv2.FILLED)
        cv2.putText(frame, text,
                    tuple(origin.astype(int)),
                    font, scale, color, thickness)
        return text_size, baseline

    def draw_object_detection_roi(self, frame, roi):
        # Draw person ROI border
        cv2.rectangle(frame,
                      tuple(roi.position), tuple(roi.position + roi.size),
                      (0, 220, 0), 2)

    def draw_object_detection_roi_with_label(self, frame, roi):
        # Draw ROI border
        self.draw_object_detection_roi(frame, roi)
        
        # Draw label box and write label
        text_scale = 0.4
        font = cv2.FONT_HERSHEY_SIMPLEX
        text_size = cv2.getTextSize("H1", font, text_scale, 1)
        line_height = np.array([0, text_size[0][1]])
        text = f"{roi.label}"
        
        self.draw_text_with_background(frame, text,
                                       roi.position - line_height * 0.4,
                                       font, scale=text_scale)

    def draw_detection_keypoints(self, frame, roi, attributes):
        keypoints = [attributes.top_color_point,
                     attributes.top_color_point]

        for point in keypoints:
            center = roi.position + roi.size * point
            cv2.circle(frame, tuple(center.astype(int)), 2, (0, 255, 255), 2)    
    
    def draw_colored_segmentation_mask(self, frame, mask):
        segments_image = frame.copy()
        aggregated_mask = np.ones(frame.shape[:2], dtype=np.uint8)
        aggregated_colored_mask = mask
        black = np.zeros(3, dtype=np.uint8)

        # Fill the area occupied by all instances with a colored instances mask image.
        cv2.bitwise_and(segments_image, black, dst=segments_image, mask=aggregated_mask)
        cv2.bitwise_or(segments_image, aggregated_colored_mask, dst=segments_image, mask=aggregated_mask)
        # Blend original image with the one, where instances are colored.
        # As a result instances masks become transparent.
        cv2.addWeighted(frame, (1-self.segmentation_weight), segments_image, self.segmentation_weight, 0, dst=frame)

    def draw_boolean_segmentation_mask(self, frame, mask, mask_color):
        segments_image = frame.copy()
        aggregated_mask = np.zeros(frame.shape[:2], dtype=np.uint8)
        aggregated_colored_mask = np.zeros(frame.shape, dtype=np.uint8)
        black = np.zeros(3, dtype=np.uint8)

        mask_color = mask_color.tolist()
        cv2.bitwise_or(aggregated_mask, mask, dst=aggregated_mask)
        cv2.bitwise_or(aggregated_colored_mask, np.asarray(mask_color, dtype=np.uint8),
                       dst=aggregated_colored_mask, mask=mask)

        # Fill the area occupied by all instances with a colored instances mask image.
        cv2.bitwise_and(segments_image, black, dst=segments_image, mask=aggregated_mask)
        cv2.bitwise_or(segments_image, aggregated_colored_mask, dst=segments_image, mask=aggregated_mask)
        # Blend original image with the one, where instances are colored.
        # As a result instances masks become transparent.
        cv2.addWeighted(frame, (1-self.segmentation_weight), segments_image, self.segmentation_weight, 0, dst=frame)


    def draw_detections(self, frame, detections):
        # self.draw_colored_segmentation_mask(frame, detections[1])
        for object_instance in detections[2]:
            self.draw_object_detection_roi_with_label(frame, object_instance)
            self.draw_boolean_segmentation_mask(frame, object_instance.mask, object_instance.color)
        
        for roi, attributes in zip(*detections[0]):
            self.draw_object_detection_roi_with_label(frame, roi)
            self.draw_detection_keypoints(frame, roi, attributes)
           
    def message_thread(self):
        global currentCar
        global currentPersons
        global currentAnimal
        global video_finished_flag
        global img_name
        global frame
        global device_id
        print("-------------------in message publish thread-------------------\n")
        while True:
            time.sleep(2)
            print("messge thread called")
            current_img=img_name
            cv2.imwrite('shared/{}'.format(img_name),frame)
            time.sleep(1)
          
    def draw_status(self, frame, detections):
        origin = np.array([10, 10])
        color = (10, 160, 10)
        font = cv2.FONT_HERSHEY_SIMPLEX
        text_scale = 0.5
        text_size, _ = self.draw_text_with_background(frame,
                                                      "Frame time: %.3fs" % (self.frame_time),
                                                      origin, font, text_scale, color)
        self.draw_text_with_background(frame,
                                       "FPS: %.1f" % (self.fps),
                                       (origin + (0, text_size[1] * 1.5)), font, text_scale, color)

        log.debug('Frame: %s/%s, detections: %s, ' \
                  'frame time: %.3fs, fps: %.1f' % \
                     (self.frame_num, self.frame_count, len(detections[-1]), self.frame_time, self.fps))
        
        if self.print_perf_stats:
            log.info('Performance stats:')
            log.info(self.frame_processor.get_performance_stats())

    def display_interactive_window(self, frame):
        color = (255, 255, 255)
        font = cv2.FONT_HERSHEY_SIMPLEX
        text_scale = 0.5
        text = "Press '%s' key to exit" % (self.BREAK_KEY_LABELS)
        thickness = 2
        text_size = cv2.getTextSize(text, font, text_scale, thickness)
        origin = np.array([frame.shape[-2] - text_size[0][0] - 10, 10])
        # line_height = np.array([0, text_size[0][1]]) * 1.5
        cv2.putText(frame, text,
                    tuple(origin.astype(int)), font, text_scale, color, thickness)
        
    def should_stop_display(self):
        key = cv2.waitKey(self.frame_timeout) & 0xFF
        return key in self.BREAK_KEYS

    def process(self, input_stream, output_stream):
        self.input_stream = input_stream
        self.output_stream = output_stream
        self.video_fps=self.input_stream.get(cv2.CAP_PROP_FPS)
        self.total_video_duration=self.frame_count/self.video_fps
        global currentCar
        global currentPersons
        global currentAnimal
        global video_finished_flag
        global img_name
        global frame
        global device_id
        prevPerson=0
        prevCar=0
        prevAnimal=0
        self.frame_num==0
        set_no=0
        prev_start_time=0
        while self.input_stream.isOpened():
            person=0
            car=0
            animal=0
            has_frame, frame = self.input_stream.read()
            if not has_frame:
                break
            if self.input_crop is not None:
                frame = Visualizer.center_crop(frame, self.input_crop)
            self.current_duration=self.frame_num/self.video_fps
            cv2.putText(frame, "Duration:{}".format(self.current_duration), (15, 100), cv2.FONT_HERSHEY_COMPLEX, 0.8, (0, 0, 255), 2)
            detections = self.frame_processor.process(frame)
            self.draw_detections(frame, detections)
            self.draw_status(frame, detections)
            for obj in detections[2]:
                #print(obj.label) # class-result-label
                if obj.label=='person':
                    person+=1
                if obj.label=='car':
                    car+=1
                if obj.label=='dog':
                    animal+=1
            
            currentPersons=person
            currentCar=car
            currentAnimal=animal
            img_name="object_detection_{}.jpg".format(int(time.time()))

            if currentPersons!=prevPerson or currentCar!=prevCar or currentAnimal!=prevAnimal:
                log.info("Persons:{}      Cars:{}       Animals:{} ".format(currentPersons,currentCar,currentAnimal))
                
            allAttributes = detections[0][1]
            # [Result1, Result2, Result3 , ... ]
            # Result1.attributes = {has_hat: 0.5, is_male: 0.1, ...}
            peopleWithHelmets = 0

            numOfPeople = len(allAttributes)
            for person in allAttributes:
                if person.attributes['has_hat'] > 0.5:
                    peopleWithHelmets += 1
                
            if peopleWithHelmets < numOfPeople:
                log.warning("PEOPLE WITHOUT HELMET")
            
            if self.output_stream:
                self.output_stream.write(frame)
            if self.display:
                self.display_interactive_window(frame)
                if self.should_stop_display():
                    break
            
            self.update_fps()
            self.frame_num += 1
        video_finished_flag=1

    @staticmethod
    def center_crop(frame, crop_size):
        fh, fw, fc = frame.shape
        crop_size[0] = min(fw, crop_size[0])
        crop_size[1] = min(fh, crop_size[1])
        return frame[(fh - crop_size[1]) // 2 : (fh + crop_size[1]) // 2,
                     (fw - crop_size[0]) // 2 : (fw + crop_size[0]) // 2,
                     :]

    def run(self, args):
        serial_no=0
        time.sleep(1)
        start_new_thread(self.message_thread,())
        print(" --------------------message publish thread started----------------------------\n")
        
        while True:
            input_stream = Visualizer.open_input_stream(args.input)
            if input_stream is None or not input_stream.isOpened():
                log.error("Cannot open input stream: %s" % args.input)
            fps = input_stream.get(cv2.CAP_PROP_FPS)
            frame_size = (int(input_stream.get(cv2.CAP_PROP_FRAME_WIDTH)),
                        int(input_stream.get(cv2.CAP_PROP_FRAME_HEIGHT)))
            self.frame_count = int(input_stream.get(cv2.CAP_PROP_FRAME_COUNT))
            if args.crop_width and args.crop_height:
                crop_size = (args.crop_width, args.crop_height)
                frame_size = tuple(np.minimum(frame_size, crop_size))
            log.info("Input stream info: %d x %d @ %.2f FPS" % \
                (frame_size[0], frame_size[1], fps))
            if args.output == "":
                args.output = "output/{}-inferenced.mp4".format("cam" if args.input == "0" else args.input.split("/")[-1][:-4])
            output_stream = Visualizer.open_output_stream(args.output, fps, frame_size)

            self.process(input_stream, output_stream)

            # Release resources
            if output_stream:
                output_stream.release()
            if input_stream:
                input_stream.release()

            cv2.destroyAllWindows()
            
    @staticmethod
    def open_input_stream(path):
        log.info("Reading input data from '%s'" % (path))
        stream = path
        try:
            stream = int(path)
        except ValueError:
            pass
        return cv2.VideoCapture(stream)
    
    @staticmethod
    def open_output_stream(path, fps, frame_size):
        output_stream = None
        if path != "":
            if not path.endswith('.avi'):
                log.warning("Output file extension is not 'avi'. " \
                        "Some issues with output can occur, check logs.")
            log.info("Writing output to '%s'" % (path))
            fourcc = cv2.VideoWriter.fourcc(*'h264')
            output_stream = cv2.VideoWriter(path,
                                            fourcc, fps, frame_size)
        return output_stream
    
def main():
    args = build_argparser().parse_args()

    log.basicConfig(format="[ %(levelname)s ] %(asctime)-15s %(message)s",
                    level=log.INFO if not args.verbose else log.DEBUG, stream=sys.stdout)

    log.debug(str(args))

    visualizer = Visualizer(args)
    visualizer.run(args)


if __name__ == '__main__':
    main()
