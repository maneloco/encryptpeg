from typing import List, Tuple
import random
import numpy as np
from convert import load_and_split, pad_to_multiple_of_8, process_channel, merge_and_save, iter_blocks_8x8
from dct import to_frequencies, from_frequencies
from encrypt import xor_encryption, xor_decryption

EMBED_ROW, EMBED_COL = 4, 4   # posición donde se guarda el bit en cada bloque

# El offset se guarda en 9 bits (no 8) porque representa el número de bits
# de la clave, y ese número puede llegar hasta 256 (clave de 32 caracteres
# * 8 bits). Con 8 bits solo llegaríamos a 255 y tendríamos overflow
# silencioso justo en ese caso límite.
OFFSET_BITS = 9


def bytearray_to_bits(data: bytearray) -> List[int]:
    bits = []
    for byte in data:
        for i in range(7, -1, -1):
            bits.append((byte >> i) & 1)
    return bits

def bits_to_bytearray(bits: List[int]) -> bytearray:
    result = bytearray()
    for i in range(0, len(bits), 8):
        byte = 0
        for j in range(8):
            byte = (byte << 1) | bits[i + j]
        result.append(byte)
    return result


# Paso de cuantización: cuanto más grande, más "sobrevive" el bit al
# redondeo de píxeles a uint8 y a la recompresión JPEG, a costa de más
# distorsión visual en el canal de crominancia. Con step=1 (par/impar)
# el bit se pierde en ~1 de cada 3 casos solo por redondear a uint8;
# con step=8 sobrevive de forma fiable en pruebas.
QUANT_STEP = 32

def embed_bit(coef: float, bit: int, step: int = QUANT_STEP) -> float:
    val = int(round(coef / step)) * step
    if (val // step) % 2 != bit:
        val += step if bit == 1 else -step
    if val == 0:
        val = step if bit == 1 else 2 * step
    return float(val)

def extract_bit(coef: float, step: int = QUANT_STEP) -> int:
    return abs(int(round(coef / step))) % 2


def resolve_offset(key_bytes: bytearray, offset: int = None) -> int:
    """
    Calcula el offset final (en bits) que se usará para reservar espacio a
    la clave. El offset SIEMPRE tiene que ser un múltiplo del tamaño natural
    de la clave en bits (8 * nº de caracteres), porque para llenar ese
    espacio se repite la clave completa las veces que hagan falta -nunca
    bits sueltos- y así el XOR cíclico sigue desencriptando correctamente.

    - Si offset es None: se elige aleatoriamente un nº de repeticiones
      válido (entre 1 y el máximo que quepa en 256 bits).
    - Si offset se especifica (0-256): se redondea hacia arriba al múltiplo
      del tamaño de la clave más cercano que sea >= offset.
    """
    natural_offset = 8 * len(key_bytes)
    if natural_offset == 0:
        raise ValueError("La clave no puede estar vacía.")

    if offset is None:
        max_repeats = max(1, 256 // natural_offset)
        repeats = random.randint(1, max_repeats)
    else:
        if not (0 <= offset <= 256):
            raise ValueError("El offset debe estar entre 0 y 256.")
        repeats = max(1, -(-offset // natural_offset))  # ceil division

    return natural_offset * repeats


def embed_message_in_channel(
    freqs_blocks: List[List[List[float]]],
    encrypted_bytes: bytearray,
    key_bytes: bytearray,
    offset: int
) -> List[List[List[float]]]:
    """
    offset: nº de bits reservados para la clave, ya resuelto con
    resolve_offset(). Debe ser múltiplo de 8 * len(key_bytes).
    """
    total_blocks = len(freqs_blocks)

    key_bits      = bytearray_to_bits(key_bytes)
    message_bits  = bytearray_to_bits(encrypted_bytes)
    natural_offset = len(key_bits)

    if offset % natural_offset != 0:
        raise ValueError(
            f"offset ({offset}) debe ser múltiplo del tamaño de la clave "
            f"en bits ({natural_offset}). Usa resolve_offset() para calcularlo."
        )
    if offset > (1 << OFFSET_BITS) - 1:
        raise ValueError(
            f"offset={offset} supera el máximo representable con "
            f"{OFFSET_BITS} bits ({(1 << OFFSET_BITS) - 1})."
        )

    repeats       = offset // natural_offset
    key_bits_full = key_bits * repeats       # repite la clave ENTERA, nunca bits sueltos
    key_bits_rev  = key_bits_full[::-1]

    offset_bits   = [(offset >> (OFFSET_BITS - 1 - i)) & 1 for i in range(OFFSET_BITS)]

    bloques_necesarios = offset + len(message_bits) + OFFSET_BITS
    if bloques_necesarios > total_blocks:
        raise ValueError(
            f"Mensaje demasiado largo: necesita {bloques_necesarios} bloques "
            f"pero el canal solo tiene {total_blocks}."
        )

    result = [block for block in freqs_blocks]

    for i, bit in enumerate(key_bits_rev):
        result[i][EMBED_ROW][EMBED_COL] = embed_bit(
            result[i][EMBED_ROW][EMBED_COL], bit
        )

    for i, bit in enumerate(message_bits):
        idx = offset + i
        result[idx][EMBED_ROW][EMBED_COL] = embed_bit(
            result[idx][EMBED_ROW][EMBED_COL], bit
        )

    for i, bit in enumerate(offset_bits):
        idx = total_blocks - OFFSET_BITS + i
        result[idx][EMBED_ROW][EMBED_COL] = embed_bit(
            result[idx][EMBED_ROW][EMBED_COL], bit
        )

    return result


def extract_message_from_channel(
    freqs_blocks: List[List[List[float]]]
) -> Tuple[bytearray, bytearray, int]:

    total_blocks = len(freqs_blocks)

    offset_bits = [
        extract_bit(freqs_blocks[total_blocks - OFFSET_BITS + i][EMBED_ROW][EMBED_COL])
        for i in range(OFFSET_BITS)
    ]
    offset = 0
    for bit in offset_bits:
        offset = (offset << 1) | bit

    key_bits_rev = [
        extract_bit(freqs_blocks[i][EMBED_ROW][EMBED_COL])
        for i in range(offset)
    ]
    key_bits  = key_bits_rev[::-1]
    key_bytes = bits_to_bytearray(key_bits)

    message_bits = [
        extract_bit(freqs_blocks[offset + i][EMBED_ROW][EMBED_COL])
        for i in range(total_blocks - OFFSET_BITS - offset)
    ]
    message_bits = message_bits[: (len(message_bits) // 8) * 8]
    encrypted_bytes = bits_to_bytearray(message_bits)

    return encrypted_bytes, key_bytes, offset


def decode_message_from_image(
    image_path: str,
    channel: str = "Cr"
) -> Tuple[str, str, int]:
    """
    Carga una imagen (BGR con OpenCV), la convierte a YCrCb, calcula las
    convoluciones (DCT) por bloques de 8x8 sobre el canal indicado
    ('Cr' o 'Cb'), extrae de la posición [EMBED_ROW][EMBED_COL] de cada
    bloque el offset (en los últimos OFFSET_BITS bloques), la clave
    (guardada en bits invertidos justo antes del offset) y el mensaje
    cifrado (a partir del offset). Desencripta con XOR usando la clave
    recuperada y devuelve (mensaje, clave, offset).
    """
    if channel not in ("Cr", "Cb"):
        raise ValueError("channel debe ser 'Cr' o 'Cb'")

    ycrcb, Y, Cr, Cb = load_and_split(image_path)
    canal = Cr if channel == "Cr" else Cb

    canal_pad = pad_to_multiple_of_8(canal)
    blocks = extract_all_blocks(canal_pad)
    freqs = [to_frequencies(b) for b in blocks]

    encrypted_bytes, key_bytes, offset = extract_message_from_channel(freqs)
    mensaje, clave = xor_decryption(encrypted_bytes, key_bytes)
    mensaje = mensaje.split('\0')[0]

    return mensaje, clave, offset




def save_image_with_message(
    image_path: str,
    message: str,
    key: str,
    output_path: str,
    offset: int = None
) -> int:
    """
    offset: nº de bits deseados para reservar a la clave (0-256). Si es
    None, se elige uno aleatorio válido. Se resuelve una única vez y se usa
    igual en los canales Cr y Cb para que ambos queden coherentes.

    Devuelve el offset final realmente usado (puede diferir ligeramente del
    solicitado: se redondea hacia arriba al múltiplo del tamaño de la clave).
    """
    from encrypt import xor_encryption

    ycrcb, Y, Cr, Cb = load_and_split(image_path)
    h_orig, w_orig = Y.shape

    Y_pad  = pad_to_multiple_of_8(Y)
    Cr_pad = pad_to_multiple_of_8(Cr)
    Cb_pad = pad_to_multiple_of_8(Cb)

    encrypted_bytes, key_bytes = xor_encryption(message + '\0', key)
    used_offset = resolve_offset(key_bytes, offset)

    Y_out = process_channel(Y_pad, lambda b: from_frequencies(to_frequencies(b)))

    Cr_blocks = extract_all_blocks(Cr_pad)
    Cr_freqs  = [to_frequencies(b) for b in Cr_blocks]
    Cr_freqs  = embed_message_in_channel(Cr_freqs, encrypted_bytes, key_bytes, used_offset)
    Cr_out    = reconstruct_channel(Cr_freqs, Cr_pad.shape)

    Cb_blocks = extract_all_blocks(Cb_pad)
    Cb_freqs  = [to_frequencies(b) for b in Cb_blocks]
    Cb_freqs  = embed_message_in_channel(Cb_freqs, encrypted_bytes, key_bytes, used_offset)
    Cb_out    = reconstruct_channel(Cb_freqs, Cb_pad.shape)

    Y_out  = Y_out[:h_orig, :w_orig]
    Cr_out = Cr_out[:h_orig, :w_orig]
    Cb_out = Cb_out[:h_orig, :w_orig]

    merge_and_save(Y_out, Cr_out, Cb_out, output_path)
    print(f"Imagen guardada en {output_path} (offset usado: {used_offset})")

    return used_offset


def extract_all_blocks(channel: np.ndarray) -> List[List[List[float]]]:
    blocks = []
    channel_f = channel.astype(np.float64)
    for y, x, block in iter_blocks_8x8(channel_f):
        blocks.append(block.tolist())
    return blocks

def reconstruct_channel(
    freqs_blocks: List[List[List[float]]],
    shape: tuple
) -> np.ndarray:
    result = np.zeros(shape, dtype=np.float64)
    h, w = shape
    idx = 0
    for y in range(0, h, 8):
        for x in range(0, w, 8):
            block = from_frequencies(freqs_blocks[idx])
            result[y:y+8, x:x+8] = np.array(block)
            idx += 1
    return np.clip(result, 0, 255).astype(np.uint8)

if __name__ == "__main__":
    # Guardar (offset=None -> se elige uno aleatorio válido)
    save_image_with_message(
        image_path="foto.jpeg",
        message="Oh...Estás haciendo magea",
        key="clave",
        output_path="foto_con_mensaje.jpeg",
        offset=None
    )

    # Extraer y desencriptar
    mensaje, clave, offset = decode_message_from_image("foto_con_mensaje.jpeg", channel="Cr")

    print(f"Mensaje recuperado: {mensaje}")
    print(f"Clave recuperada:   {clave}")
    print(f"Offset usado:       {offset}")
