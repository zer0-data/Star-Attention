# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

try:
    from .modeling_llama import LlamaForCausalLM
except (ImportError, AttributeError):
    # transformers >=4.46 removed LlamaFlashAttention2 as a standalone class.
    # modeling_llama.py was written against 4.44.x and is incompatible with
    # newer versions. Qwen3 (requires >=4.51) cannot coexist with the old
    # Llama file on the same install. Llama models need transformers==4.44.x.
    LlamaForCausalLM = None

from .modeling_qwen3 import Qwen3ForCausalLM
