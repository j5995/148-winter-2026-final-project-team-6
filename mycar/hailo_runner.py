import cv2
import numpy as np
from donkeycar.utils import throttle as compute_throttle
from hailo_platform import (
    HEF,
    VDevice,
    HailoStreamInterface,
    InferVStreams,
    ConfigureParams,
    InputVStreamParams,
    OutputVStreamParams,
    FormatType
)

class HailoModelRunner:
    def __init__(self, hef_path):
        self.hef_path = hef_path

        self.device = VDevice()
        self.hef = HEF(hef_path)

        self.configure_params = ConfigureParams.create_from_hef(hef=self.hef, interface=HailoStreamInterface.PCIe)
        network_groups = self.device.configure(self.hef, self.configure_params)

        self.network_group = network_groups[0]
        self.network_group_params = self.network_group.create_params()

        self.input_vstreams_params = InputVStreamParams.make(self.network_group, format_type=FormatType.UINT8)
        self.output_vstreams_params = OutputVStreamParams.make(self.network_group, format_type=FormatType.FLOAT32)

        self.input_vstream_info = self.hef.get_input_vstream_infos()[0]
        self.output_vstream_infos = self.hef.get_output_vstream_infos()

        self.image_height, self.image_width, self.channels = self.input_vstream_info.shape
        self.input_batch = np.empty((1, self.image_height, self.image_width, self.channels), dtype=np.uint8)

        # Persistent pipeline — created once, reused every frame
        self.infer_pipeline = InferVStreams(self.network_group, self.input_vstreams_params, self.output_vstreams_params)
        self.infer_pipeline.__enter__()

        self.activated = self.network_group.activate(self.network_group_params)
        self.activated.__enter__()

        # Pre-create the input dict to avoid allocation per frame
        self.input_data = {self.input_vstream_info.name: self.input_batch}

        # Determine output format once — never changes between frames
        self.has_two_outputs = len(self.output_vstream_infos) >= 2

    def run(self, image):
        image = cv2.resize(image, (self.image_width, self.image_height))
        self.input_batch[0] = image
        infer_results = self.infer_pipeline.infer(self.input_data)

        if self.has_two_outputs:
            steering = float(infer_results[self.output_vstream_infos[0].name][0, 0])
            throttle = float(infer_results[self.output_vstream_infos[1].name][0, 0])
        else:
            steering = float(infer_results[self.output_vstream_infos[0].name][0, 0])
            throttle = compute_throttle(steering)

        return steering, throttle

    def __del__(self):
        try:
            if hasattr(self, 'activated'):
                self.activated.__exit__(None, None, None)
            if hasattr(self, 'infer_pipeline'):
                self.infer_pipeline.__exit__(None, None, None)
            if hasattr(self, 'device'):
                self.device.release()
        except Exception as e:
            print(f"[HailoModelRunner] Warning during cleanup: {e}")
