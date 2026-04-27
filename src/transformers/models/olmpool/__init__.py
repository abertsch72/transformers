# Copyright 2025 the HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from typing import TYPE_CHECKING

from ...utils import _LazyModule
from ...utils.import_utils import define_import_structure


if TYPE_CHECKING:
    from .configuration_olmpool import *
    from .modeling_olmpool import *
else:
    import sys

    _file = globals()["__file__"]
    sys.modules[__name__] = _LazyModule(__name__, _file, define_import_structure(_file), module_spec=__spec__)

    # Register aliases so HF auto-mapping can resolve model_type names to this module.
    # HF convention maps model_type "foo_bar" -> transformers.models.foo_bar, but these
    # model types all live in the single olmpool module.
    _lazy_mod = sys.modules[__name__]
    for _alias in [
        "transformers.models.olmo3_preorder",
        "transformers.models.olmo3_preorder_headwise_qknorm",
        "transformers.models.olmo2_headwise_qknorm",
        "transformers.models.llama3_qknorm",
        "transformers.models.llama3_reordered",
    ]:
        sys.modules.setdefault(_alias, _lazy_mod)
    del _lazy_mod, _alias
