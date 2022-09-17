# Copyright 2019 Yan Yan
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

from .overwrite_spconv.write_spconv2 import register_spconv2

try:
    import spconv
except ImportError:
    IS_SPCONV2_AVAILABLE = False
else:
    if hasattr(spconv, '__version__') and spconv.__version__ >= '2.0.0':
        IS_SPCONV2_AVAILABLE = register_spconv2()
    else:
        IS_SPCONV2_AVAILABLE = False

__all__ = ['IS_SPCONV2_AVAILABLE']

