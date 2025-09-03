import pandas as pd
import joblib
import sys
import numpy as np # numpy도 필요하므로 추가

# --- DummyModel 클래스 정의 (모델 로드를 위해 필요) ---
# prepare_dummy_data.py에 있는 것과 동일해야 합니다.
class DummyModel:
    def predict_proba(self, X):
        n_samples = X.shape[0]
        random_probs = np.random.rand(n_samples) # 0과 1 사이의 무작위 값
        return np.array([[1 - p, p] for p in random_probs])

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)

# --- predict_bosch_failure_with_threshold 함수 정의 ---
def predict_bosch_failure_with_threshold(data_path, model_path, threshold=0.5):
    """
    테스트 데이터를 로드하여 불량을 예측하고, 예측에 기여한 주요 특성을 반환합니다.
    분류 임계값을 조절하여 Precision/Recall을 제어할 수 있습니다.

    Args:
        data_path (str): 테스트 데이터 파일의 경로 (예: CSV).
        model_path (str): 학습된 모델 파일의 경로 (예: .joblib).
        threshold (float): 불량으로 분류할 확률 임계값 (0.0 ~ 1.0). 기본값은 0.5.
    Returns:
        list: 예측 결과 리스트 (각 항목은 딕셔너리).
    """
    print(f"\n--- 임계값 {threshold}로 예측 수행 중 ---")
    print(f"데이터 로드 중: {data_path}")
    try:
        # 테스트 데이터 로드
        # prepare_dummy_data.py에서 이미 전처리된 데이터를 저장했으므로, 추가 전처리 없음
        test_data = pd.read_csv(data_path)
    except FileNotFoundError:
        print(f"오류: 테스트 데이터 파일 '{data_path}'을(를) 찾을 수 없습니다.")
        return []

    print(f"모델 로드 중: {model_path}")
    try:
        # 학습된 모델 로드
        model = joblib.load(model_path)
    except FileNotFoundError:
        print(f"오류: 모델 파일 '{model_path}'을(를) 찾을 수 없습니다.")
        return []
    except Exception as e:
        print(f"오류: 모델 로드 중 문제가 발생했습니다: {e}")
        return []

    print("불량 확률 예측 중...")
    # 모델의 predict_proba 메서드를 사용하여 불량일 확률을 얻습니다.
    # DummyModel은 [정상일 확률, 불량일 확률] 형태의 배열을 반환하므로, 불량일 확률(인덱스 1)을 사용합니다.
    prediction_proba = model.predict_proba(test_data)[:, 1]

    # 임계값을 이용하여 최종 불량 여부를 결정합니다.
    # 예측 확률이 임계값 이상이면 1(불량), 미만이면 0(정상)으로 분류합니다.
    prediction = (prediction_proba >= threshold).astype(int)

    results = []
    # 각 샘플에 대한 예측 결과와 확률을 저장합니다.
    for i, (pred, proba) in enumerate(zip(prediction, prediction_proba)):
        status = "불량" if pred == 1 else "정상"
        results.append({
            "샘플_인덱스": i,
            "예측_결과": status,
            "불량_확률": f"{proba:.4f}",
            "사용된_임계값": threshold
        })
    return results

# --- 시뮬레이션 실행 부분 ---
if __name__ == "__main__":
    # prepare_dummy_data.py에서 생성된 파일 경로를 사용합니다.
    simulated_test_data_path = '/workspaces/newwww/simulated_test_data.csv'
    dummy_model_path = '/workspaces/newwww/dummy_bosch_model.joblib'

    print("\n--- 생산 엔지니어를 위한 불량 예측 및 임계값 조절 시연 시작 ---")

    # 1. 임계값 0.5 (기본값)으로 예측
    # 불량 확률이 50% 이상일 때 불량으로 판단합니다.
    # 이는 일반적으로 Precision과 Recall의 균형을 맞추는 시작점입니다.
    results_0_5 = predict_bosch_failure_with_threshold(simulated_test_data_path, dummy_model_path, threshold=0.5)
    print("\n[임계값 0.5 예측 결과 (일부)]:")
    for i, res in enumerate(results_0_5):
        if i < 5: # 상위 5개 결과만 출력하여 간결하게 보여줍니다.
            print(res)
        else:
            break

    # 2. 임계값 0.3 (Recall 증가 목적)으로 예측
    # 불량 확률이 30%만 넘어도 불량으로 판단합니다.
    # 이는 불량품을 놓치는 것(False Negative)의 비용이 매우 클 때 사용됩니다.
    # 더 많은 샘플이 '불량'으로 분류될 수 있으며, 이는 재현율을 높이는 데 기여합니다.
    results_0_3 = predict_bosch_failure_with_threshold(simulated_test_data_path, dummy_model_path, threshold=0.3)
    print("\n[임계값 0.3 예측 결과 (일부)]:")
    for i, res in enumerate(results_0_3):
        if i < 5: # 상위 5개 결과만 출력
            print(res)
        else:
            break

    # 3. 임계값 0.7 (Precision 증가 목적)으로 예측
    # 불량 확률이 70% 이상일 때만 불량으로 판단합니다.
    # 이는 정상품을 불량으로 오인하는 것(False Positive)의 비용이 매우 클 때 사용됩니다.
    # 더 적은 샘플이 '불량'으로 분류될 수 있으며, 이는 정밀도를 높이는 데 기여합니다.
    results_0_7 = predict_bosch_failure_with_threshold(simulated_test_data_path, dummy_model_path, threshold=0.7)
    print("\n[임계값 0.7 예측 결과 (일부)]:")
    for i, res in enumerate(results_0_7):
        if i < 5: # 상위 5개 결과만 출력
            print(res)
        else:
            break

    print("\n--- 시연 완료 ---")
    print("생산 엔지니어는 이처럼 'threshold' 값을 조절하여")
    print("불량품을 놓치지 않는 것(Recall)과 불량 예측의 정확도(Precision) 사이의 균형을 맞출 수 있습니다.")
    print("최적의 임계값은 불량품이 발생했을 때의 비용과 정상품을 오인했을 때의 비용을 고려하여 결정해야 합니다.")

    sys.exit(0)