# LightGlue ile GPS-Denied Video-Orthofoto Eslestirme Notlari

Bu klasorde kurulacak sistemin hedefi, yalnizca goruntu tabanli olarak video karelerini ortofotoya eslestirip dunya koordinatinda konum uretmektir. Tasarim kararlari asagidaki resmi kaynaklara dayandirildi:

- LightGlue resmi repo: https://github.com/cvg/LightGlue
- LightGlue ICCV 2023 paper: https://arxiv.org/abs/2306.13643
- hloc resmi repo: https://github.com/cvg/Hierarchical-Localization
- AnyLoc proje sayfasi: https://anyloc.github.io/
- FoundLoc paper: https://anyloc.github.io/FoundLoc/assets/FoundLoc_draft.pdf

## Bu kaynaklar bize ne soyluyor

1. LightGlue bir "ince eslestirme" motorudur.
   LightGlue dogrudan "haritada neredeyim" sorusunu cozmez. Iki aday goruntu arasinda hizli ve guclu nokta eslestirme yapar. Bu nedenle GPS-denied konumlandirmada LightGlue'in onune mutlaka bir kaba aday secim asamasi koymak gerekir.

2. hloc hiyerarsik lokalizasyon fikrini netlestiriyor.
   hloc boru hatti kabaca su sekildedir:
   - once retrieval ile aday referanslari bul
   - sonra local feature extraction + matching yap
   - en son geometri ile dogrula

3. AnyLoc ve FoundLoc, GPS-denied sahada kaba aramanin neden gerekli oldugunu gosteriyor.
   Ozellikle FoundLoc, GNSS-denied UAV localization icin VIO/VPR benzeri kaba ankraj + ince eslestirme yaklasiminin pratikte calistigini gosteriyor.

4. Bizim haritamiz kucuk oldugu icin ilk surumde retrieval yerine exhaustive tile search kullanilabilir.
   Bu projedeki ortofoto sehir olceginde degil. Bu nedenle ilk guvenilir surumde tum haritayi cok sayida tile'a bolup:
   - once tum tile'lar icinde arama
   - sonra lokal takipte sadece yakin komsular icinde arama
   yapmak mantikli ve uygulanabilir.

5. Homography bizim problem icin temel geometri dogrulama aracidir.
   Nadir veya nadire yakin goruntu ile ortofoto arasinda lokal planar varsayim cogu karede yeterlidir. Bu nedenle LightGlue match cikislarini `cv2.findHomography(..., USAC_MAGSAC)` ile dogrulayip:
   - inlier sayisi
   - inlier orani
   - reprojection hatasi
   - frame footprint'inin tile icinde mantikli konuma dusmesi
   ile puanlamak en dogru ilk tercih.

## Bu klasorde kurulacak yeni mimari

1. `georeferenced_orthophoto_map.py`
   GeoTIFF'i okuyacak, piksel-latlon donusumlerini yapacak ve tile kesebilecek.

2. `lightglue_feature_matcher.py`
   LightGlue + secilebilir extractor katmani.
   Ilk varsayilan extractor `aliked` olarak secildi.
   Neden:
   - learned local feature oldugu icin aerial-video vs orthofoto farkinda daha guclu baslangic verdi
   - BSD-3-Clause lisanslidir
   - tek kare smoke testte bu veri uzerinde `localized` sonuc uretti
   `sift` hala yedek secenek olarak korunuyor.

3. `orthophoto_tile_index.py`
   Orthofotoyu birden fazla tile boyutunda parcalayacak ve aday listeleyecek.

4. `gps_denied_video_orthophoto_localizer.py`
   Her video karesi icin:
   - global veya local tile adaylari sececek
   - LightGlue ile eslestirecek
   - homography ile dogrulayacak
   - frame merkezini ortofotoya projekte edip lat/lon uretecek

5. `run_lightglue_video_to_orthophoto.py`
   Tek giris noktasi olacak.

6. `export_flight_logs_to_csv.py`
   `logs/` altindaki ArduPilot loglarini ayiklayip `log_csv/` klasorune yazar.
   Boylece GPS, IMU, AHR2, ATT, CAM, POS ve XKF kanallari ayrik CSV olarak kullanilabilir.

## Ilk surumde bilerek neleri yapmiyoruz

- GPS'e runtime prior olarak guvenmiyoruz
- Kalman filtre kullanmiyoruz
- epipolar geometry'yi ana karar mekanizmasi yapmiyoruz
- buyuk retrieval modeli zorunlu kilmiyoruz

## LightGlue klasorunde hazir olan log kanallari

`log_csv/00000003_birinci_ucus` ve `log_csv/00000004_ikinci_ucus` altinda artik su kanallar hazir:

- `GPS.csv`: GNSS position, speed, ground course
- `IMU.csv`: raw gyro + accelerometer
- `AHR2.csv`: fused attitude + position
- `ATT.csv`: attitude controller state
- `POS.csv`: position estimate
- `CAM.csv`: camera trigger pose
- `XKF1.csv` ... `XKF5.csv`: EKF durum ve tutarlilik kanallari

Bu veriler yeni LightGlue sistemi icin su islerde kullanilabilir:

- video/log zaman esleme
- ucus baslangici tespiti
- arama bolgesini IMU/AHR2 ile daraltma
- GPS'i sadece GT veya offline dogrulama icin kullanma
- CAM tetikleri ile frame-foto esleme

## Ilk surumden sonraki en dogru gelisim yolu

1. Global retrieval katmani ekle
   Kucuk haritada exhaustive tile search yeterli. Harita buyurse AnyLoc benzeri retrieval katmani eklenmeli.

2. Orientation hipotezleri ekle
   Zor karelerde frame'i veya tile'i belirli aci hipotezleriyle denemek faydali olabilir.

3. Yerel takipte motion prior ekle
   GNSS-denied demek IMU yasak demek degil. Gerekirse sadece IMU/VIO ile arama bolgesi daraltilabilir.

4. Cok kareli track-consistency skoru ekle
   Tek kare en iyi homography yerine kisa pencere icinde zamansal tutarlilik puani daha saglam sonuc verir.
