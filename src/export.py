
import os
import cv2
import numpy as np


def save_as_jpg(image: np.ndarray, output_path: str, quality: int = 95) -> str:
    ##Guarda una imagen (array BGR de numpy) como .jpg, forzando la extensión
    ##y la calidad de compresión.

    ##cv2.imwrite infiere el formato a partir de la extensión del archivo, así
    ##que si output_path no termina en .jpg/.jpeg, lo corregimos aquí para
    ##evitar que se guarde en un formato distinto sin darnos cuenta.

    ##quality: 0-100. Cuanto más bajo, más se comprime y más se puede degradar
    ##el mensaje embebido en los coeficientes DCT. Se recomienda 90-95 para no
    ##perder los bits guardados en steganography.py.
    
    base, ext = os.path.splitext(output_path)
    if ext.lower() not in (".jpg", ".jpeg"):
        output_path = base + ".jpg"

    ok = cv2.imwrite(
        output_path,
        image,
        [cv2.IMWRITE_JPEG_QUALITY, quality]
    )
    if not ok:
        raise IOError(f"No se pudo guardar la imagen en {output_path}")

    return output_path


def verify_saved_image(path: str) -> dict:

    ##Comprueba que el archivo guardado es un JPEG válido:
      ##- existe y no está vacío
      ##- empieza con la cabecera mágica de JPEG (FF D8 FF)
      ##- se puede reabrir con cv2 sin errores
      ##-las dimensiones son coherentes (>0)

    ##Devuelve un dict con el resultado de cada comprobación, útil para
    ##depurar antes de mandar la imagen por WhatsApp u otro canal.
    
    result = {
        "existe": False,
        "no_vacio": False,
        "cabecera_jpeg_valida": False,
        "se_puede_reabrir": False,
        "dimensiones": None,
        "errores": [],
    }

    if not os.path.isfile(path):
        result["errores"].append(f"El archivo no existe: {path}")
        return result
    result["existe"] = True

    size = os.path.getsize(path)
    if size == 0:
        result["errores"].append("El archivo está vacío (0 bytes)")
        return result
    result["no_vacio"] = True

    with open(path, "rb") as f:
        header = f.read(3)
    if header == b"\xff\xd8\xff":
        result["cabecera_jpeg_valida"] = True
    else:
        result["errores"].append(
            f"Cabecera inválida: {header!r} (se esperaba FF D8 FF)"
        )

    reopened = cv2.imread(path)
    if reopened is None:
        result["errores"].append("cv2.imread no pudo reabrir el archivo")
    else:
        result["se_puede_reabrir"] = True
        result["dimensiones"] = reopened.shape

    return result


if __name__ == "__main__":
    # Ejemplo de uso 
    ruta = save_as_jpg(cv2.imread("foto_con_mensaje.jpeg"), "foto_con_mensaje.jpg")
    diagnostico = verify_saved_image(ruta)

    print(f"Verificación de {ruta}:")
    for clave, valor in diagnostico.items():
        print(f"  {clave}: {valor}")

    if diagnostico["errores"]:
        print("\n❌ Hay problemas con el archivo guardado.")
    else:
        print("\n✅ Archivo JPEG válido, listo para enviar.")
