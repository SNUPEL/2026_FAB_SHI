"""pytest 부팅 시 MKL 단일 스레드 가드.

Windows conda의 MKL은 numpy.linalg.lstsq(LAPACK gelsd)의 내부 멀티스레딩에서 간헐적으로
native crash(faulthandler C-stack dump)를 낸다. 원인은 앱 코드가 아니라 MKL 스레딩 경합이다.
numpy가 import되기 전에 MKL을 단일 스레드/sequential로 고정해 이 경합 자체를 제거한다.
conftest.py는 test 모듈보다 먼저 로드되므로 numpy import보다 앞선다.

torch intra-op 병렬성(OMP_NUM_THREADS)은 건드리지 않는다. 사용자가 외부에서 명시한 값은 존중한다.
Windows가 아닌 환경에서는 아무 것도 하지 않는다(no-op).
"""

import os

if os.name == "nt":
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("MKL_THREADING_LAYER", "SEQUENTIAL")
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
