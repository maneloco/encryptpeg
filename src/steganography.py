from typing import List, Tuple
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


def embed_bit(coef: float, bit: int) -> float:
    val = int(round(coef))
    if val % 2 != bit:
        val += 1 if bit == 1 else -1
    if val == 0:
        val = 1 if bit == 1 else 2
    return float(val)

def extract_bit(coef: float) -> int:
    return abs(int(round(coef))) % 2


def embed_message_in_channel(
    freqs_blocks: List[List[List[float]]],
    encrypted_bytes: bytearray,
    key_bytes: bytearray
) -> List[List[List[float]]]:

    total_blocks = len(freqs_blocks)

    key_bits      = bytearray_to_bits(key_bytes)
    message_bits  = bytearray_to_bits(encrypted_bytes)
    key_bits_rev  = key_bits[::-1]

    offset        = len(key_bits_rev)
    if offset > (1 << OFFSET_BITS) - 1:
        raise ValueError(
            f"La clave es demasiado larga: offset={offset} bits, "
            f"máximo representable con {OFFSET_BITS} bits es {(1 << OFFSET_BITS) - 1}."
        )
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
) -> Tuple[bytearray, bytearray]:

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

    return encrypted_bytes, key_bytes




def save_image_with_message(
    image_path: str,
    message: str,
    key: str,
    output_path: str
):
    from encrypt import xor_encryption

    ycrcb, Y, Cr, Cb = load_and_split(image_path)
    h_orig, w_orig = Y.shape

    Y_pad  = pad_to_multiple_of_8(Y)
    Cr_pad = pad_to_multiple_of_8(Cr)
    Cb_pad = pad_to_multiple_of_8(Cb)

    encrypted_bytes, key_bytes = xor_encryption(message, key)

    Y_out = process_channel(Y_pad, lambda b: from_frequencies(to_frequencies(b)))

    Cr_blocks = extract_all_blocks(Cr_pad)
    Cr_freqs  = [to_frequencies(b) for b in Cr_blocks]
    Cr_freqs  = embed_message_in_channel(Cr_freqs, encrypted_bytes, key_bytes)
    Cr_out    = reconstruct_channel(Cr_freqs, Cr_pad.shape)

    Cb_blocks = extract_all_blocks(Cb_pad)
    Cb_freqs  = [to_frequencies(b) for b in Cb_blocks]
    Cb_freqs  = embed_message_in_channel(Cb_freqs, encrypted_bytes, key_bytes)
    Cb_out    = reconstruct_channel(Cb_freqs, Cb_pad.shape)

    Y_out  = Y_out[:h_orig, :w_orig]
    Cr_out = Cr_out[:h_orig, :w_orig]
    Cb_out = Cb_out[:h_orig, :w_orig]

    merge_and_save(Y_out, Cr_out, Cb_out, output_path)
    print(f"Imagen guardada en {output_path}")


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
    # Guardar
    save_image_with_message(
        image_path="foto.jpeg",
        message="Oh...Estás haciendo magea",
        key="clave",
        output_path="foto_con_mensaje.jpeg"
    )

    # Extraer y desencriptar
    ycrcb, Y, Cr, Cb = load_and_split("foto_con_mensaje.jpeg")
    h, w = Cr.shape
    Cr_pad = pad_to_multiple_of_8(Cr)
    Cr_blocks = extract_all_blocks(Cr_pad)
    Cr_freqs  = [to_frequencies(b) for b in Cr_blocks]

    encrypted_bytes, key_bytes = extract_message_from_channel(Cr_freqs)
    mensaje, clave = xor_decryption(encrypted_bytes, key_bytes)

    print(f"Mensaje recuperado: {mensaje}")
    print(f"Clave recuperada:   {clave}")