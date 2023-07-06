import os

# Mostrar solo errores de TensorFlow
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
# Desabilitar GPU ( correr en CPU )
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
from alpr.alpr import ALPR
from argparse import ArgumentParser
import yaml
import logging
from timeit import default_timer as timer
import cv2
import time
from collections import OrderedDict
import pytesseract
import pandas as pd
#pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
import boto3
import re
logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
from twilio.rest import Client
from dotenv import load_dotenv
load_dotenv()



def whatsapp_alert( msj=''):

    if msj =='':
        return False
    
    to_          = os.getenv('whatsapp_number')
    from_        = os.getenv('whatsapp_number_twilio')
    
    account_sid = os.getenv('whastapp_account_sid')
    auth_token  = os.getenv('whastapp_auth_token')
    

    client_twilio = Client(account_sid, auth_token)
    message             = client_twilio.messages.create(
            body        = msj,
            from_       = f'whatsapp:{from_}', # Número de Twilio para enviar mensajes de WhatsApp
            to          = f'whatsapp:{to_}'
        )
    

def main_demo(cfg, demo=True, benchmark=True, save_vid=False):

    logger.info(f'Leyendo archivo encargos...')
    encargos = pd.read_csv('patentes_encargos.csv')
    encargos_list = encargos['PPU'].tolist()
    encargos_list = [elem.strip() for elem in encargos_list]
    logger.info(f'Cargadas: {len(encargos_list)} patentes ')


    alpr = ALPR(cfg['modelo'])
    video_path = cfg['video']['fuente']
    #video_path = 0
    default_width= 640
    default_height = 480
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, default_width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, default_height)


    logger.info(f'Se va analizar la fuente: {video_path}')
    # Cada cuantos frames hacer inferencia
    cv2.CAP_DSHOW = True
    patentes_detectadas = OrderedDict()
    max_patentes = 10
    avg = 0
    while True:
        return_value, frame = cap.read()
        if return_value:
            #frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        else:
            # Descomenten esto para camara IP Esto es por si el stream deja de transmitir algún
            # frame o se tarda más de lo normal. En este caso simplemente volvemos a intentar leer el frame.
            continue
        
        frame_w_pred, avg,patente_actual,roi = alpr.mostrar_predicts(frame)

        if patente_actual == '':
            continue

        if avg < cfg['modelo']['confianza_avg_ocr']:
            continue

        if patente_actual in patentes_detectadas and time.time() - patentes_detectadas[patente_actual] < 120:
            print("La patente {} ya se ha procesado en los últimos 2 minutos.".format(patente_actual))
            continue
        else:
            patentes_detectadas[patente_actual] = time.time()

        if cfg['modelo']['patente_en_csv']:

            if patente_actual in encargos_list :
                print('****** PATENTE CON ENCARGO *********')
                print('****** '+patente_actual+' *********')

                if(cfg['modelo']['whatsapp']):
                    whatsapp_alert('Patente con encargo : '+patente_actual)
        
        else:
            if patente_actual not in encargos_list :
                print('****** PATENTE NO ENCONTRADA EN LISTADO  *********')
                print('****** '+patente_actual+' *********')
        
        if len(patentes_detectadas) > max_patentes:
            patentes_detectadas.popitem(last=False)  # Elimina la primera patente ingresada

    


if __name__ == '__main__':
    try:
        parser = ArgumentParser()
        parser.add_argument("--cfg", dest="cfg_file", help="Path del archivo de config, \
                            default: ./config.yaml", default='config.yaml')
        parser.add_argument("--demo", dest="demo",
                            action='store_true', help="En vez de guardar las patentes, mostrar las predicciones")
       
        parser.add_argument("--benchmark", dest="bench",
                            action='store_true', help="Medir la inferencia (incluye todo el pre/post processing")
        args = parser.parse_args()
        with open(args.cfg_file, 'r') as stream:
            try:
                cfg = yaml.safe_load(stream)
            except yaml.YAMLError as exc:
                logger.exception(exc)
        main_demo(cfg, args.demo, args.bench)
    except Exception as e:
        logger.exception(e)