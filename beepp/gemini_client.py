#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : gemini_client.py
# Author : Xiaolin Fang
# Email  : fxlfang@gmail.com
# Date   : 06/28/2025
#
# Distributed under terms of the MIT license.

"""
Gemini drop-in replacement for gpt_client.py.

Install: pip install google-genai opencv-python numpy
Auth:    export GEMINI_API_KEY="your-key"   (picked up automatically by the SDK)
"""

import cv2
import numpy as np
from google import genai
from google.genai import types
from beepp.utils.common_utils import plot_images

# Note: Alternatives - "gemini-2.5-pro", "gemini-robotics-er-1.6-preview"
GEMINI_MODEL = "gemini-2.5-flash-lite"


def extract_tag_content(content: str, tag: str):
    """
    Extract the content of a tag from a string.
    Args:
        content:    str. The content string
        tag:        str. The tag to extract

    Returns:
        str. The content of the tag, or None if not found.
    """
    start = content.find(f"<{tag}>")
    end = content.find(f"</{tag}>")
    if start == -1 or end == -1:
        return None
    return content[start + len(tag) + 2: end]


def _image_parts(image_sequence: list[np.ndarray]) -> list[types.Part]:
    """
    Encode a list of BGR numpy images as JPEG bytes and wrap each as a Part.
    The images are converted RGB->BGR before encoding (matching the GPT client).
    """
    parts = []
    for img in image_sequence:
        success, buf = cv2.imencode(".jpg", img[..., ::-1])  # RGB -> BGR for cv2
        if not success:
            raise RuntimeError("cv2.imencode failed")
        parts.append(
            types.Part.from_bytes(data=buf.tobytes(), mime_type="image/jpeg")
        )
    return parts


class GeminiClient:
    def __init__(self, verbose_level=0):
        # GEMINI_API_KEY (or GOOGLE_API_KEY) is read from the environment automatically
        self.client = genai.Client()
        self.verbose_level = verbose_level
        self.model = GEMINI_MODEL

    def _generate(self, contents: list) -> str:
        """Call generate_content and return the response text."""
        response = self.client.models.generate_content(
            model=self.model,
            contents=contents,
        )
        if self.verbose_level > 0:
            print('=' * 10, 'Gemini Returned Result', '=' * 10)
            print(response.text)
            print('=' * 45)
        return contents, response

    # ------------------------------------------------------------------
    # Naming
    # ------------------------------------------------------------------

    def obtain_Gemini_naming(self, image_sequence: list[np.ndarray]):
        """
        Prompt Gemini to obtain the name of an object.
        Args:
            image_sequence: list of RGB numpy images
        Returns:
            (contents, response) — mirrors the GPT version's (messages, result)
        """
        instruction = (
            "I have provided a few images. "
            "Please return the name for the object in each image.\n"
            "Your output should be multiple lines, each line corresponding to an image "
            "of the format <output>{object_name}</output>."
            "If an image contains multiple objectss, please only return the name of the "
            "most salient object."
            "Do not return anything else."
        )

        # Gemini contents: list of Parts (images first, then text)
        contents = [
            *_image_parts(image_sequence),
            types.Part.from_text(text=instruction),
        ]

        return self._generate(contents)

    def obtain_gemini_naming_auto(self, image_sequence: list[np.ndarray], max_retries=3):
        for i in range(max_retries):
            try:
                contents, result = self.obtain_Gemini_naming(image_sequence)
                output_lines = result.text.strip().split('\n')
                object_names = [extract_tag_content(line, "output") for line in output_lines]
                object_names = [n for n in object_names if n is not None]
                if len(object_names) != len(image_sequence):
                    raise RuntimeError(
                        f'Gemini Result Parsing Error. '
                        f'Received {len(object_names)} results for {len(image_sequence)} queries.'
                    )
                return object_names
            except Exception as e:
                print(f"Error in Gemini call. Retrying. Error: {e}")
        raise RuntimeError("Gemini call failed.")

    # ------------------------------------------------------------------
    # Most-likely object selection
    # ------------------------------------------------------------------

    def obtain_mostlikelyobject(
        self,
        image_sequence: list[np.ndarray],
        text_desc: str,
        allow_none=True,
    ):
        """
        Prompt Gemini to select the image that best matches a text description.
        Args:
            image_sequence: list of RGB numpy images
            text_desc:      target description string
            allow_none:     if True, model may return ID 0 (no match)
        Returns:
            (contents, response)
        """
        instruction = "I have provided you a sequence of images.\n"
        instruction += f"Please select the ID of the image that is most similar to the description '{text_desc}'.\n"
        instruction += "Put your answer in the format of <output>{ID}</output>\n"
        instruction += "Let's think step by step following the pattern:\n"
        instruction += "image_1_mask = '...' # Describe the mask in the first image\n"
        instruction += "image_2_mask = '...' # Describe the mask in the second image\n"
        instruction += "...\n"
        instruction += "output: <output>{ID}</output>  # return the image ID that matches the text description in the last line.\n"
        if allow_none:
            instruction += "If None of them matches the text description, return ID 0.\n"
        instruction += "Do not include anything else in the last line other than <output>{ID}</output>"

        contents = [
            *_image_parts(image_sequence),
            types.Part.from_text(text=instruction),
        ]

        return self._generate(contents)

    def obtain_mostlikelyobject_auto(
        self,
        image_sequence: list[np.ndarray],
        text_desc: str,
        max_retries=3,
        allow_none=True,
        vis=False
    ):
        if vis:
            plot_images(image_sequence)
        min_valid_id = 0 if allow_none else 1
        for i in range(max_retries):
            try:
                contents, result = self.obtain_mostlikelyobject(image_sequence, text_desc, allow_none)
                last_line = result.text.strip().split('\n')[-1]
                selected_image_id = int(extract_tag_content(last_line, "output"))
                if selected_image_id < min_valid_id or selected_image_id > len(image_sequence):
                    raise RuntimeError('Gemini Result Parsing Error.')
                return selected_image_id - 1  # convert 1-based -> 0-based index
            except Exception as e:
                print(f"Error in Gemini call. Retrying. Error: {e}")
        raise RuntimeError("Gemini call failed.")


# ------------------------------------------------------------------
# Tests (mirror the originals)
# ------------------------------------------------------------------

def test_gemini_naming():
    import pickle
    gemini_client = GeminiClient()
    with open('../example/captures.pkl', 'rb') as f:
        data = pickle.load(f)
    rgb_images = [data['rgb']]
    object_names = gemini_client.obtain_gemini_naming_auto(rgb_images)
    print(f'Object names: {object_names}')
    plot_images(rgb_images, object_names)


if __name__ == '__main__':
    test_gemini_naming()