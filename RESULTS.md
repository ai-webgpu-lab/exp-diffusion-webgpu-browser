# Results

## 1. 실험 요약
- 저장소: exp-diffusion-webgpu-browser
- 커밋 해시: 624d07f
- 실험 일시: 2026-05-20T15:41:19.620Z -> 2026-05-20T15:41:23.919Z
- 담당자: ai-webgpu-lab
- 실험 유형: `multimodal`
- 상태: `success`

## 2. 질문
- 브라우저 diffusion 실험으로 넘기기 전에 sec per image, steps per sec, fail-rate 보고 경로를 먼저 고정할 수 있는가
- prompt tag, scheduler, seed, resolution, fallback metadata가 diffusion 결과 문서에 같이 남는가
- 실제 browser diffusion runtime 교체 전 deterministic prompt-to-image harness로 반복 검증이 가능한가

## 3. 실행 환경
### 브라우저
- 이름: Chrome
- 버전: 147.0.7727.15

### 운영체제
- OS: Linux
- 버전: unknown

### 디바이스
- 장치명: Linux x86_64
- device class: `desktop-high`
- CPU: 16 threads
- 메모리: 32 GB
- 전원 상태: `unknown`

### GPU / 실행 모드
- adapter: navigator.gpu available
- backend: `webgpu`
- fallback triggered: `false`
- worker mode: `worker`
- cache state: `warm`
- required features: ["shader-f16"]
- limits snapshot: {"maxStorageBuffersPerShaderStage":8,"maxTextureDimension2D":8192}

## 4. 워크로드 정의
- 시나리오 이름: Diffusion Browser Readiness
- 입력 프로필: 768x512-28-steps
- 데이터 크기: promptTag=observatory-aurora-v1; scheduler=dpmpp-2m-karras; resolution=768x512; seed=41; steps=28; previews=4; backend=webgpu; fallback=false; safety=pass; automation=playwright-chromium, promptTag=observatory-aurora-v1; scheduler=dpmpp-2m-karras; resolution=768x512; seed=41; steps=28; previews=4; backend=webgpu; fallback=false; safety=pass; realAdapter=fallback(adapter.loadModel is not a function); automation=playwright-chromium
- dataset: diffusion-fixture-v1
- model_id 또는 renderer: deterministic-diffusion-browser-v1
- 양자화/정밀도: -
- resolution: -
- context_tokens: -
- output_tokens: -

## 5. 측정 지표
### 공통
- time_to_interactive_ms: 757.4 ~ 1508 ms
- init_ms: 590 ms
- success_rate: 1
- peak_memory_note: 32 GB reported by browser
- error_type: -

### Diffusion
- sec_per_image: 0.59 s
- steps_per_sec: 68.29
- resolution_success_rate: 1
- oom_or_fail_rate: 0
- worker modes: worker
- backends: webgpu
- fallback states: false

## 6. 결과 표
| Run | Scenario | Backend | Cache | Mean | P95 | Notes |
|---|---|---:|---:|---:|---:|---|
| 1 | Diffusion Browser Readiness | webgpu | warm | 68.29 | 0.59 | resolution_success=1, oom_or_fail=0 |
| 2 | Diffusion Browser Readiness | webgpu | warm | 68.29 | 0.59 | resolution_success=1, oom_or_fail=0 |

## 7. 관찰
- browser diffusion readiness baseline은 backend=webgpu, fallback_triggered=false, worker_mode=worker로 기록됐다.
- diffusion summary는 sec_per_image=0.59, steps_per_sec=68.29, oom_or_fail_rate=0였다.
- diffusion metadata는 promptTag=observatory-aurora-v1; scheduler=dpmpp-2m-karras; resolution=768x512; seed=41; steps=28; previews=4; backend=webgpu; fallback=false; safety=pass; automation=playwright-chromium로 남았다.
- playwright-chromium로 수집된 automation baseline이며 headless=true, browser=Chromium 147.0.7727.15.
- 실제 runtime/model/renderer 교체 전 deterministic harness 결과이므로, 절대 성능보다 보고 경로와 재현성 확인에 우선 의미가 있다.

## 8. Real Adapter vs Deterministic
- adapter: real=diffusion-xenova-sd-turbo-300, deterministic=deterministic-mock
- adapter_run: real=connected, deterministic=deterministic
- success_rate: real=1, deterministic=1

## 9. 결론
- browser diffusion readiness harness가 prompt-to-image sec_per_image, steps_per_sec, resolution success, fail rate를 같은 문서에 남기게 됐다.
- 다음 단계는 deterministic canvas surface를 실제 browser diffusion runtime, UNet, VAE, scheduler path로 교체하되 sec_per_image/steps_per_sec/resolution_success_rate/oom_or_fail_rate metric 구조를 유지하는 것이다.
- 이후 `bench-diffusion-browser-shootout`와 `app-browser-image-lab`의 공통 diffusion fixture 입력으로 재사용할 수 있다.

## 10. 첨부
- 스크린샷: ./reports/screenshots/01-diffusion-browser-readiness.png, ./reports/screenshots/10-diffusion-webgpu-browser-real-diffusion.png
- 로그 파일: ./reports/logs/01-diffusion-browser-readiness.log, ./reports/logs/10-diffusion-webgpu-browser-real-diffusion.log
- raw json: ./reports/raw/01-diffusion-browser-readiness.json, ./reports/raw/10-diffusion-webgpu-browser-real-diffusion.json
- 배포 URL: https://ai-webgpu-lab.github.io/exp-diffusion-webgpu-browser/
- 관련 이슈/PR: -
