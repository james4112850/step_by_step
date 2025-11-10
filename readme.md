

## 結構

  
<img width="331" height="112" alt="image" src="https://github.com/user-attachments/assets/d9726407-47d8-435d-9c55-b273e29a730f" />


- **step_a_preprocess1.py**: 圖像初步預處理。
  
  <img width="321" height="131" alt="image" src="https://github.com/user-attachments/assets/dbb7e261-8020-47f7-b727-aa9d6b3543a4" />

- **step_b_crop_car.py**: 從圖像中裁剪車輛。
  
  <img width="315" height="137" alt="image" src="https://github.com/user-attachments/assets/82a856ef-e7cb-49cd-81f5-5e29058b4805" />

- **step_c_preprocess2.py**: 進一步圖像預處理。
    
  <img width="313" height="138" alt="image" src="https://github.com/user-attachments/assets/8c57824f-ecac-4e6d-9e20-0bf690ab9b17" />

- **step_d_crop_plate.py**: 從車輛圖像中裁剪車牌。
    
  <img width="326" height="135" alt="image" src="https://github.com/user-attachments/assets/bf0baffd-2d80-434b-bf79-4dab60678443" />

- **step_e_characters.py**: 分割車牌字元的步驟。
  
  <img width="407" height="132" alt="image" src="https://github.com/user-attachments/assets/b1684ae1-68a8-4076-a2c5-e97d12467e09" />

- **utils.py**: 函式庫(utilitiy functions)。


## 使用指南
1. 運行`step_a_preprocess1.py`進行初步預處理。
2. 使用`step_b_crop_car.py`裁剪車輛。
3. 執行`step_c_preprocess2.py`進一步預處理。
4. 使用`step_d_crop_plate.py`裁剪車牌。
5. 使用`step_e_characters.py`進行字元分割。


