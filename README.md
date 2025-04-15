# CDCGAN Report

본 저장소는 **MNIST 기반 조건부 딥 컨볼루션 생성적 적대 신경망 (Conditional Deep Convolutional GAN, CDCGAN)** 구현 및 실험 결과를 포함합니다.  
이 프로젝트는 *국민대학교 비주얼컴퓨팅 최신기술 (2025-1학기)* 과제로 수행되었습니다.

---

> **작성자**  
> 국민대학교 나노전자물리학과 & 소프트웨어전공  
> 20201914 박규민

---

## 보고서 보기

- [보고서 PDF 열기](./20201914.pdf)

---

## 주요 내용

- CDCGAN 구조 설계 (Generator, Discriminator, Conditioning Mechanism)
- PyTorch Lightning 기반 학습
- Gradio 인터페이스를 통한 조건부 이미지 생성
- Latent vector와 label에 따른 이미지 생성 결과 분석
