# Hair20K & UV Latent Texture Database 구조 안내

---

## Hair20K 데이터베이스 위치

### NAS (공유 저장소)
- 경로: `/Database1/hair20k/*/.data`
- 설명: 원본 DB `.data` 파일만 존재 (전처리 전 상태)

### 150번 서버 (접속 불가로 미확인)
- 예상 경로: 미확인, tonghs 디렉토리 어딘가..
- 설명: `.data` → `.npz` 변환된 DB가 존재할 것으로 추정됨

### 45번 서버 (교수님 방, VR컴)
- 경로: `/hdd_sda1/tonghs/DATA/hair20k`
- 구성:
  - `.data`: 원본 DB
  - `.npz`: 전처리된 DB (.data → .npz)

---

## UV Latent Texture 데이터베이스

### 150번 서버 (접속 불가로 미확인)
- 데이터 존재 가능성 있음 (추정)

### 45번 서버
- 경로:
  - `/hdd_sda1/tonghs/workspace/SparseHair/StrandVAE/output/shape_texture/2024-10-25_01-16-09`
  - `/hdd_sda1/tonghs/workspace/SparseHair/StrandVAE/output/shape_texture/2024-10-28_23-47-24`
- 설명: StrandVAE로 추출된 UV latent texture 결과물

---

## 관련 스크립트 파일 (`script/`)

| 파일명 | 설명 |
|--------|------|
| `train_strand_vae.sh` | StrandVAE 모델 학습 스크립트 |
| `get_shape_texture.sh` | Pretrained StrandVAE로 UV latent texture 생성 |
| `train_hairdiffmae.sh` | UV latent texture DB로 HairDiffMAE 학습 |

---

## 참고
- 150 서버 vscode ssh 접속이 안되는데, 150 서버 리눅스 버전 오래됨이 이슈인 것 같음. 가능하면 20.04 로 업그레이드 요망.