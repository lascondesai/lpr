from timeit import default_timer as timer

import cv2
import numpy as np

from .detector import PlateDetector
from .ocr import PlateOCR
import re


class ALPR():
    def __init__(self, cfg: dict):
        
        input_size = cfg['resolucion_detector']
        if input_size not in (384, 512, 608):
            raise ValueError('Modelo detector no existe! Opciones { 384, 512, 608 }')
        
        detector_path = f'alpr/models/detection/tf-yolo_tiny_v4-{input_size}x{input_size}-custom-anchors/'
        self.detector = PlateDetector(detector_path, input_size, score=cfg['confianza_detector'])
        self.ocr = PlateOCR(cfg['numero_modelo_ocr'], cfg['confianza_avg_ocr'], cfg['confianza_low_ocr'])

    def predict(self, frame: np.ndarray) -> list:
        """
        Devuelve todas las patentes reconocidas
        a partir de un frame.

        Parametros:
            frame: np.ndarray sin procesar (Colores en orden: RGB)
        Returns:
            Una lista con todas las patentes reconocidas
        """
        # Preprocess
        input_img = self.detector.preprocess(frame)
        # Inference
        yolo_out = self.detector.predict(input_img)
        # Bounding Boxes despues de NMS
        bboxes = self.detector.procesar_salida_yolo(yolo_out)
        # Hacer OCR a cada patente localizada
        iter_coords = self.detector.yield_coords(frame, bboxes)
        patentes = self.ocr.predict(iter_coords, frame)
        if self.guardar_bd:
            self.update_in_memory(patentes)
        return patentes
    
    
    def mostrar_predicts(self, frame: np.ndarray):

        """
        Mostrar localizador + reconocedor

        Parametros:
            frame: np.ndarray sin procesar (Colores en orden: RGB)
        Returns:
            frame con el bounding box de la patente y
            la prediccion del texto de la patente

            total_time: tiempo de inferencia sin contar el dibujo
            de los rectangulos
        """
        avg         = 0
        subcadena   =''
        plate       = False
        # Preprocess
        input_img = self.detector.preprocess(frame)
        # Inference
        yolo_out = self.detector.predict(input_img)
        # Bounding Boxes despues de NMS
        bboxes = self.detector.procesar_salida_yolo(yolo_out)

        
        # Hacer y mostrar OCR
        iter_coords = self.detector.yield_coords(frame, bboxes)
        fontScale   = 1.25
        roi         = []


        for yolo_prediction in iter_coords:
            
            x1, y1, x2, y2, _ = yolo_prediction
            
            width       = x2 - x1
            height      = y2 - y1
            width_new   = int(width * 1.4)
            height_new  = int(height * 1)
            x1_new      = x1 - int((width_new - width) / 2)
            y1_new      = y1 - int((height_new - height) / 2)
            x2_new      = x2 + int((width_new - width) / 2)
            y2_new      = y2 + int((height_new - height) / 2)

            cv2.rectangle(frame, (x1_new, y1_new), (x2_new, y2_new), (255, 0, 0), 2)
            
            plate           = None
            plate, probs    = self.ocr.predict_ocr(x1_new, y1_new, x2_new, y2_new, frame)
            avg             = round(np.mean(probs)*100,2)
            subcadena       = (''.join(plate)+' '+str(avg)+'%')
            plate           = (''.join(plate))
            
            
            if avg > self.ocr.confianza_avg and self.ocr.none_low(probs, thresh=self.ocr.none_low_thresh):

                roi = frame[y1_new:y2_new, x1_new:x2_new]

                cv2.putText(img=frame, text=subcadena, org=(x1 - 20, y1 - 15),
                            fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=fontScale,
                            color=[0, 0, 0], lineType=cv2.LINE_AA, thickness=6)
                
                cv2.putText(img=frame, text=subcadena, org=(x1 - 20, y1 - 15),
                            fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=fontScale,
                            color=[255, 255, 255], lineType=cv2.LINE_AA, thickness=2)
                
                image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                # Guardar la imagen en formato RGB
                
                cv2.imwrite('./tmp/'+plate+'.jpg',image_rgb)

                return frame,avg,plate,roi

                
        return frame,avg,'',roi
            
