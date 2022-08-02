import cv2
import statistics
import numpy as np
import pandas as pd
from matplotlib import pyplot
import face_mesh_matsumoto
import os
import gc

#-------------------------------------
#画像から顔のランドマーク検出を行う関数
#-------------------------------------
def face_landmark_facemesh(img, frame_count, video_name, interval, step):

    facemesh = face_mesh_matsumoto.Facemesh(0.7, 0.5) #facemeshを呼びだす
    results = facemesh.run(img) #入ってきたフレーム画像に対してfacemeshを行う

    if results == None: #顔検出ができていなかったら
        #RGBすべての要素をNAN値にする
        R, G, B, rg, rb, gb, R_r, G_r, B_r, rg_r, rb_r, gb_r, R_l, G_l, B_l, rg_l, rb_l, gb_l, R_a, G_a, B_a, rg_a, rb_a, gb_a = "NAN"

    else: #顔検出出来ていたら
        nose_img = results["nose"] #鼻部画像の切り出し
        r_cheek_img = results["rcheek"] #右頬画像の切り出し
        l_cheek_img = results["lcheek"] #左頬画像の切り出し

        height_nose, weight_nose, channal_nose = nose_img.shape #鼻部の画像サイズ等の取り出し
        height_r_cheek, weight_r_cheek, channal_r_cheek = r_cheek_img.shape #右頬の画像サイズ等の取り出し
        height_l_cheek, weight_l_cheek, channal_l_cheek = l_cheek_img.shape #左頬の画像サイズ等の取り出し

        #配列にする
        nose_img_array = np.asarray(nose_img)
        r_cheek_img_array = np.asarray(r_cheek_img)
        l_cheek_img_array = np.asarray(l_cheek_img)

        file_number = ((765 - interval) // (step - 1)) + 2 #何個のデータができるか計算する

        for number in range(file_number): #全ての範囲に対して処理を行う
            r_value_list = [] #R成分値のリスト
            g_value_list = [] #G成分値のリスト
            b_value_list = [] #B成分値のリスト
            rb_value_list = [] #R-B成分値のリスト
            rg_value_list = [] #R-G成分値のリスト
            gb_value_list = [] #G-B成分値のリスト

            R_r_value_list = [] #R成分値のリスト
            G_r_value_list = [] #G成分値のリスト
            B_r_value_list = [] #B成分値のリスト
            rb_r_value_list = [] #R-B成分値のリスト
            rg_r_value_list = [] #R-G成分値のリスト
            gb_r_value_list = [] #G-B成分値のリスト

            R_l_value_list = [] #R成分値のリスト
            G_l_value_list = [] #G成分値のリスト
            B_l_value_list = [] #B成分値のリスト
            rb_l_value_list = [] #R-B成分値のリスト
            rg_l_value_list = [] #R-G成分値のリスト
            gb_l_value_list = [] #G-B成分値のリスト

            if number == 0: #最初だったら
                #csvファイルを追記モードで開く
                f = open(".//result//csv_result//" + str(video_name) + "//" + str(interval) + "_" + str(step) + "//" + str(int(step) * int(number)) + "_" + str(int(interval) - 1) +"_landmark.csv", "a")
                
                #--------------------------
                #鼻部のデータ取得・成形
                #--------------------------
                for i in range(0, height_nose):
                    for j in range(0, weight_nose):
                        rgb_value = int(nose_img_array[i, j, :][0]) + int(nose_img_array[i, j, :][1]) + int(nose_img_array[i, j, :][2])
                        if (rgb_value >= step * number) & (rgb_value <= interval - 1):
                            b_value = int(nose_img_array[i, j, :][0]) #B成分値の取得
                            g_value = int(nose_img_array[i, j, :][1]) #G成分値の取得
                            r_value = int(nose_img_array[i, j, :][2]) #R成分値の取得

                            rb_value = r_value - b_value #該当ピクセルのR-B
                            rg_value = r_value - g_value #該当ピクセルのR-G
                            gb_value = g_value - b_value #該当ピクセルのG-B

                            #リストに値を保存する
                            r_value_list.append(r_value)
                            g_value_list.append(g_value)
                            b_value_list.append(b_value)
                            rb_value_list.append(rb_value)
                            rg_value_list.append(rg_value)
                            gb_value_list.append(gb_value)

                #--------------------------
                #右頬のデータ取得・成形
                #--------------------------
                #取り出した右頬を1ピクセルずつ見ていく
                for i in range(0, height_r_cheek):
                    for j in range(0, weight_r_cheek):
                        r_rgb_value = int(r_cheek_img_array[i, j, :][0]) + int(r_cheek_img_array[i, j, :][1]) + int(r_cheek_img_array[i, j, :][2])
                        if (r_rgb_value >= step * number) & (r_rgb_value <= interval - 1):
                            B_r_value = int(r_cheek_img_array[i, j, :][0]) #B成分値の取得
                            G_r_value = int(r_cheek_img_array[i, j, :][1]) #G成分値の取得
                            R_r_value = int(r_cheek_img_array[i, j, :][2]) #R成分値の取得

                            rb_r_value = R_r_value - B_r_value #該当ピクセルのR-B
                            rg_r_value = R_r_value - G_r_value #該当ピクセルのR-G
                            gb_r_value = G_r_value - B_r_value #該当ピクセルのG-B

                            #値をリストに保存する
                            R_r_value_list.append(R_r_value)
                            G_r_value_list.append(G_r_value)
                            B_r_value_list.append(B_r_value)
                            rb_r_value_list.append(rb_r_value)
                            rg_r_value_list.append(rg_r_value)
                            gb_r_value_list.append(gb_r_value)

                #--------------------------
                #左頬のデータ取得・成形
                #--------------------------
                #左頬を1ピクセルずつ見ていく
                for i in range(0, height_l_cheek):
                    for j in range(0, weight_l_cheek):
                        l_rgb_value = int(l_cheek_img_array[i, j, :][0]) + int(l_cheek_img_array[i, j, :][1]) + int(l_cheek_img_array[i, j, :][2])
                        if (l_rgb_value >= step * number) & (l_rgb_value <= interval - 1):
                            B_l_value = int(l_cheek_img_array[i, j, :][0]) #B成分値の取得
                            G_l_value = int(l_cheek_img_array[i, j, :][1]) #G成分値の取得
                            R_l_value = int(l_cheek_img_array[i, j, :][2]) #R成分値の取得

                            rb_l_value = R_l_value - B_l_value #該当ピクセルのR-B
                            rg_l_value = R_l_value - G_l_value #該当ピクセルのR-G
                            gb_l_value = G_l_value - B_l_value #該当ピクセルのG-B

                            #値をリストに保存する
                            R_l_value_list.append(R_l_value)
                            G_l_value_list.append(G_l_value)
                            B_l_value_list.append(B_l_value)
                            rb_l_value_list.append(rb_l_value)
                            rg_l_value_list.append(rg_l_value)
                            gb_l_value_list.append(gb_l_value)


            else: #stepを使うようになったら
                #csvファイルの作成
                f = open(".//result//csv_result//" + str(video_name) + "//" + str(interval) + "_" + str(step) + "//" + str(int(step) * int(number)) + "_" + str(int(interval) - 1 + int(step) * int(number)) +"_landmark.csv", "a") #新規作成モードでファイルを開く
                #--------------------------
                #鼻部のデータ取得・成形
                #--------------------------                
                for i in range(0, height_nose):
                    for j in range(0, weight_nose):
                        rgb_value = int(nose_img_array[i, j, :][0]) + int(nose_img_array[i, j, :][1]) + int(nose_img_array[i, j, :][2])
                        if (rgb_value >= step * number) & (rgb_value <= interval - 1 + step * number):
                            b_value = int(nose_img_array[i, j, :][0]) #B成分値の取得
                            g_value = int(nose_img_array[i, j, :][1]) #G成分値の取得
                            r_value = int(nose_img_array[i, j, :][2]) #R成分値の取得

                            rb_value = r_value - b_value #該当ピクセルのR-B
                            rg_value = r_value - g_value #該当ピクセルのR-G
                            gb_value = g_value - b_value #該当ピクセルのG-B

                            #リストに値を保存する
                            r_value_list.append(r_value)
                            g_value_list.append(g_value)
                            b_value_list.append(b_value)
                            rb_value_list.append(rb_value)
                            rg_value_list.append(rg_value)
                            gb_value_list.append(gb_value)

                #--------------------------
                #右頬のデータ取得・成形
                #--------------------------
                #取り出した右頬を1ピクセルずつ見ていく
                for i in range(0, height_r_cheek):
                    for j in range(0, weight_r_cheek):
                        r_rgb_value = int(r_cheek_img_array[i, j, :][0]) + int(r_cheek_img_array[i, j, :][1]) + int(r_cheek_img_array[i, j, :][2])
                        if (r_rgb_value >= step * number) & (r_rgb_value <= interval - 1 + step * number):
                            B_r_value = int(r_cheek_img_array[i, j, :][0]) #B成分値の取得
                            G_r_value = int(r_cheek_img_array[i, j, :][1]) #G成分値の取得
                            R_r_value = int(r_cheek_img_array[i, j, :][2]) #R成分値の取得

                            rb_r_value = R_r_value - B_r_value #該当ピクセルのR-B
                            rg_r_value = R_r_value - G_r_value #該当ピクセルのR-G
                            gb_r_value = G_r_value - B_r_value #該当ピクセルのG-B

                            #値をリストに保存する
                            R_r_value_list.append(R_r_value)
                            G_r_value_list.append(G_r_value)
                            B_r_value_list.append(B_r_value)
                            rb_r_value_list.append(rb_r_value)
                            rg_r_value_list.append(rg_r_value)
                            gb_r_value_list.append(gb_r_value)

                #--------------------------
                #左頬のデータ取得・成形
                #--------------------------
                #左頬を1ピクセルずつ見ていく
                for i in range(0, height_l_cheek):
                    for j in range(0, weight_l_cheek):
                        l_rgb_value = int(l_cheek_img_array[i, j, :][0]) + int(l_cheek_img_array[i, j, :][1]) + int(l_cheek_img_array[i, j, :][2])
                        if (l_rgb_value >= step * number) & (l_rgb_value <= interval + step * number):
                            B_l_value = int(l_cheek_img_array[i, j, :][0]) #B成分値の取得
                            G_l_value = int(l_cheek_img_array[i, j, :][1]) #G成分値の取得
                            R_l_value = int(l_cheek_img_array[i, j, :][2]) #R成分値の取得

                            rb_l_value = R_l_value - B_l_value #該当ピクセルのR-B
                            rg_l_value = R_l_value - G_l_value #該当ピクセルのR-G
                            gb_l_value = G_l_value - B_l_value #該当ピクセルのG-B

                            #値をリストに保存する
                            R_l_value_list.append(R_l_value)
                            G_l_value_list.append(G_l_value)
                            B_l_value_list.append(B_l_value)
                            rb_l_value_list.append(rb_l_value)
                            rg_l_value_list.append(rg_l_value)
                            gb_l_value_list.append(gb_l_value)


            #RGB値の平均値を取得するため,一次元化して平均する
            if len(r_value_list) == 0: #リストに何も入っていない=その範囲のピクセルが存在しなかったとき
                #すべての値をNANにする
                R = "NAN"
                G = "NAN"
                B = "NAN"
                rb = "NAN"
                rg = "NAN"
                gb = "NAN"
            else: #値がある時は平均値を求める
                R = statistics.mean(r_value_list)
                G = statistics.mean(g_value_list)
                B = statistics.mean(b_value_list)
                rb = statistics.mean(rb_value_list)
                rg = statistics.mean(rg_value_list)
                gb = statistics.mean(gb_value_list)


            #RGB値の平均値を取得するため,一次元化して平均する
            if len(R_r_value_list) == 0: #リストに何も入っていない=その範囲のピクセルが存在しなかったとき
                #すべての値をNANにする
                R_r = "NAN"
                G_r = "NAN"
                B_r = "NAN"
                rb_r = "NAN"
                rg_r = "NAN"
                gb_r = "NAN"
            else: #値が入っていたら
                #平均した値を求める
                R_r = statistics.mean(R_r_value_list)
                G_r = statistics.mean(G_r_value_list)
                B_r = statistics.mean(B_r_value_list)
                rb_r = statistics.mean(rb_r_value_list)
                rg_r = statistics.mean(rg_r_value_list)
                gb_r = statistics.mean(gb_r_value_list)


            #RGB値の平均値を取得するため,一次元化して平均する
            if len(R_l_value_list) == 0: #リストに何も入っていない=その範囲のピクセルが存在しなかったとき
                #すべての値をNANにする
                R_l = "NAN"
                G_l = "NAN"
                B_l = "NAN"
                rb_l = "NAN"
                rg_l = "NAN"
                gb_l = "NAN"

                #頬の左右平均値もNANにする
                R_a = "NAN"
                G_a = "NAN"
                B_a = "NAN"
                rg_a = "NAN"
                rb_a = "NAN"
                gb_a = "NAN"

            else: #リストに値が入っていたら
                #平均値を求める
                R_l = statistics.mean(R_l_value_list)
                G_l = statistics.mean(G_l_value_list)
                B_l = statistics.mean(B_l_value_list)
                rb_l = statistics.mean(rb_l_value_list)
                rg_l = statistics.mean(rg_l_value_list)
                gb_l = statistics.mean(gb_l_value_list)

                if R_r != "NAN":
                    #右頬と左頬の平均値も求める
                    R_a = (R_r + R_l) / 2
                    G_a = (G_r + G_l) / 2
                    B_a = (B_r + B_l) / 2
                    rg_a = (rg_r + rg_l) / 2
                    rb_a = (rb_r + rb_l) / 2
                    gb_a = (gb_r + gb_l) / 2

            #実験経過時間を求める
            time = frame_count

            #取得してきた値を書き出す
            f.write(str(R) + "," + str(G) + "," + str(B) + "," + str(rg) + "," + str(rb) + "," + str(gb) + "," + str(R_r) + "," + str(G_r) + "," + str(B_r) + "," + str(rg_r) + "," + str(rb_r) + "," + str(gb_r) + "," + str(R_l) + "," + str(G_l) + "," + str(B_l) + "," + str(rg_l) + "," + str(rb_l) + "," + str(gb_l) + "," + str(R_a) + "," + str(G_a) + "," + str(B_a) + "," + str(rg_a) + "," + str(rb_a) + "," + str(gb_a) + "," + str(time) + "\n")
            f.close()

            del r_value_list, g_value_list, b_value_list, rb_value_list, rg_value_list, gb_value_list, R_r_value_list, G_r_value_list, B_r_value_list, rb_r_value_list, rg_r_value_list, gb_r_value_list, R_l_value_list, G_l_value_list, B_l_value_list, rb_l_value_list, rg_l_value_list, gb_l_value_list
            gc.collect() #メモリ解放

#---------------------------------
#作り直したデータを平滑化する関数
#---------------------------------
def smooth_data(interval, step):

    file_number = ((765 - interval) // (step - 1)) + 2 #何個のデータができるか計算する
    #書き換えポイント
    for number in range(file_number):
        if number == 0:
            df = pd.read_csv(".//result//csv_result//" + str(video_name) + "//" + str(interval) + "_" + str(step) + "//" + str(int(step) * int(number)) + "_" + str(int(interval) - 1) +"_landmark.csv", encoding="utf-8") #先に作成したデータファイルを開く 
            # df = pd.read_csv(".//result//csv_result//" + str(video_name) + "//" + str(interval) + "_" + str(step) + "//" + str(int(step) * int(number)) + "_" + str(int(interval) - 1) +"_landmark.csv", encoding="shift_jis") #先に作成したデータファイルを開く
        else:
            df = pd.read_csv(".//result//csv_result//" + str(video_name) + "//" + str(interval) + "_" + str(step) + "//" + str(int(step) * int(number)) + "_" + str(int(interval) - 1 + int(step) * int(number)) +"_landmark.csv", encoding="utf-8") #先に作成したデータファイルを開く 
            # df = pd.read_csv(".//result//csv_result//" + str(video_name) + "//" + str(interval) + "_" + str(step) + "//" + str(int(step) * int(number)) + "_" + str(int(interval) - 1 + int(step) * int(number)) +"_landmark.csv", encoding="shift_jis") #先に作成したデータファイルを開く

        #線形補間をするために値をfloat型に変換する(NAN値はError扱い)
        use_data_R = pd.to_numeric(df["R Value"], errors="coerce")
        use_data_G = pd.to_numeric(df["G Value"], errors="coerce")
        use_data_B = pd.to_numeric(df["B Value"], errors="coerce")
        use_data_R_B = pd.to_numeric(df["R-B Value"], errors="coerce")
        use_data_R_G = pd.to_numeric(df["R-G Value"], errors="coerce")
        use_data_G_B = pd.to_numeric(df["G-B Value"], errors="coerce")
        use_data_bR = pd.to_numeric(df["Both R Value"], errors="coerce")
        use_data_bG = pd.to_numeric(df["Both G Value"], errors="coerce")
        use_data_bB = pd.to_numeric(df["Both B Value"], errors="coerce")
        use_data_bR_B = pd.to_numeric(df["Both R-B Value"], errors="coerce")
        use_data_bR_G = pd.to_numeric(df["Both R-G Value"], errors="coerce")
        use_data_bG_B = pd.to_numeric(df["Both G-B Value"], errors="coerce")
        use_data_time = pd.to_numeric(df["Time"])

        #float型に変換したデータを新しくuse_dataとして保存する
        use_data = pd.concat([use_data_R, use_data_G, use_data_B, use_data_R_B, use_data_R_G, use_data_G_B, use_data_bR, use_data_bG, use_data_bB, use_data_bR_B, use_data_bR_G, use_data_bG_B, use_data_time], axis=1)

        #欠損値を線形補間する
        use_data_drop_nan = use_data.interpolate()

        #鼻-頬のデータも作成
        use_data_drop_nan["nose-cheek R-B Value"] = use_data_drop_nan["R-B Value"] - use_data_drop_nan["Both R-B Value"]
        use_data_drop_nan["nose-cheek G-B Value"] = use_data_drop_nan["G-B Value"] - use_data_drop_nan["Both G-B Value"]
        use_data_drop_nan["nose-cheek R-G Value"] = use_data_drop_nan["R-G Value"] - use_data_drop_nan["Both R-G Value"]

        #スムージングする(約20秒でのスムージング)
        smooth_data = use_data_drop_nan.rolling(20).mean()
        smooth_data = smooth_data.rename(columns={"R Value":"Smotth R Value", "G Value":"Smooth G Value", "B Value":"Smooth B Value", "R-B Value":"Smooth R-B Value", "R-G Value":"Smooth R-G Value", "G-B Value":"Smooth G-B Value", "Both R Value":"Smooth Both R Value", "Both G Value":"Smooth Both G Value", "Both B Value":"Smooth Both B Value","Both R-B Value":"Smooth Both R-B Value", "Both R-G Value":"Smooth Both R-G Value", "Both G-B Value":"Smooth Both G-B Value", "nose-cheek R-B Value":"Smooth nose-cheek R-B Value", "nose-cheek G-B Value":"Smooth nose-cheek G-B Value", "nose-cheek R-G Value":"Smooth nose-cheek R-G Value"})
        #スムージングする(約5秒でのスムージング)
        smooth_data_2 = use_data_drop_nan.rolling(5).mean()
        smooth_data_2 = smooth_data_2.rename(columns={"R Value":"Smotth R Value", "G Value":"Smooth G Value", "B Value":"Smooth B Value", "R-B Value":"Smooth R-B Value", "R-G Value":"Smooth R-G Value", "G-B Value":"Smooth G-B Value", "Both R Value":"Smooth Both R Value", "Both G Value":"Smooth Both G Value", "Both B Value":"Smooth Both B Value","Both R-B Value":"Smooth Both R-B Value", "Both R-G Value":"Smooth Both R-G Value", "Both G-B Value":"Smooth Both G-B Value", "nose-cheek R-B Value":"Smooth nose-cheek R-B Value", "nose-cheek G-B Value":"Smooth nose-cheek G-B Value", "nose-cheek R-G Value":"Smooth nose-cheek R-G Value"})


        #スムージングしたデータをデータフレームに落とし込む
        analysis_data = pd.concat([use_data_drop_nan, smooth_data], axis=1)
        analysis_data_2 = pd.concat([use_data_drop_nan, smooth_data_2], axis=1)

        if number == 0:
            #csvファイルとして書き出しをする
            analysis_data.to_csv(".//result//csv_result//" + str(video_name) + "//" + str(interval) + "_" + str(step) + "//" + str(int(step) * int(number)) + "_" + str(int(interval) - 1) + "_smooth_20sec.csv", encoding="utf-8")
            analysis_data_2.to_csv(".//result//csv_result//" + str(video_name) + "//" + str(interval) + "_" + str(step) + "//" + str(int(step) * int(number)) + "_" + str(int(interval) - 1) + "_smooth_5sec.csv", encoding="utf-8")
            # analysis_data.to_csv(".//result//csv_result//" + str(video_name) + "//" + str(interval) + "_" + str(step) + "//" + str(int(step) * int(number)) + "_" + str(int(interval) - 1) + "_smooth_20sec.csv", encoding="shift_jis")
            # analysis_data_2.to_csv(".//result//csv_result//" + str(video_name) + "//" + str(interval) + "_" + str(step) + "//" + str(int(step) * int(number)) + "_" + str(int(interval) - 1) + "_smooth_5sec.csv", encoding="shift_jis")
        else:
            #csvファイルとして書き出しをする
            analysis_data.to_csv(".//result//csv_result//" + str(video_name) + "//" + str(interval) + "_" + str(step) + "//" + str(int(step) * int(number)) + "_" + str(int(interval) - 1 + int(step) * int(number)) + "_smooth_20sec.csv", encoding="utf-8")
            analysis_data_2.to_csv(".//result//csv_result//" + str(video_name) + "//" + str(interval) + "_" + str(step) + "//" + str(int(step) * int(number)) + "_" + str(int(interval) - 1 + int(step) * int(number)) + "_smooth_5sec.csv", encoding="utf-8")
            # analysis_data.to_csv(".//result//csv_result//" + str(video_name) + "//" + str(interval) + "_" + str(step) + "//" + str(int(step) * int(number)) + "_" + str(int(interval) - 1 + int(step) * int(number)) +" _smooth_20sec.csv", encoding="shift_jis")
            # analysis_data_2.to_csv(".//result//csv_result//" + str(video_name) + "//" + str(interval) + "_" + str(step) + "//" + str(int(step) * int(number)) + "_" + str(int(interval) - 1 + int(step) * int(number)) +" _smooth_5sec.csv", encoding="shift_jis")


#---------------------
#グラフを作成する関数
#---------------------
def make_graph(interval, step):

    file_number = ((765 - interval) // (step - 1)) + 2 #何個のデータができるか計算する

    for number in range(file_number):
        if number == 0:
            re_analysis_data_20second = pd.read_csv(".//result//csv_result//" + str(video_name) + "//" + str(interval) + "_" + str(step) + "//" + str(int(step) * int(number)) + "_" + str(int(interval) - 1) +"_smooth_20sec.csv", encoding="utf-8")
            re_analysis_data_5second = pd.read_csv(".//result//csv_result//" + str(video_name) + "//" + str(interval) + "_" + str(step) + "//" + str(int(step) * int(number)) + "_" + str(int(interval) - 1) +"_smooth_5sec.csv", encoding="utf-8")
            #re_analysis_data_20second = pd.read_csv(".//result//csv_result//" + str(video_name) + "//" + str(interval) + "_" + str(step) + "//" + str(int(step) * int(number)) + "_" + str(int(interval) - 1) +"_smooth_20sec.csv", encoding="shift_jis")
            #re_analysis_data_5second = pd.read_csv(".//result//csv_result//" + str(video_name) + "//" + str(interval) + "_" + str(step) + "//" + str(int(step) * int(number)) + "_" + str(int(interval) - 1) +"_smooth_5sec.csv", encoding="shift_jis")
        else:
            re_analysis_data_20second = pd.read_csv(".//result//csv_result//" + str(video_name) + "//" + str(interval) + "_" + str(step) + "//" + str(int(step) * int(number)) + "_" + str(int(interval) - 1 + int(step) * int(number)) +"_smooth_20sec.csv", encoding="utf-8")
            re_analysis_data_5second = pd.read_csv(".//result//csv_result//" + str(video_name) + "//" + str(interval) + "_" + str(step) + "//" + str(int(step) * int(number)) + "_" + str(int(interval) - 1 + int(step) * int(number)) +"_smooth_5sec.csv", encoding="utf-8")
            #re_analysis_data_20second = pd.read_csv(".//result//csv_result//" + str(video_name) + "//" + str(interval) + "_" + str(step) + "//" + str(int(step) * int(number)) + "_" + str(int(interval) - 1 + int(step) * int(number)) +"_smooth_20sec.csv", encoding="shift_jis")
            #re_analysis_data_5second = pd.read_csv(".//result//csv_result//" + str(video_name) + "//" + str(interval) + "_" + str(step) + "//" + str(int(step) * int(number)) + "_" + str(int(interval) - 1 + int(step) * int(number)) +"_smooth_5sec.csv", encoding="shift_jis")

        analysis_data_20second = pd.read_csv(".//result//csv_result//" + str(video_name) + "//" + str(video_name) + "_smooth_20sec.csv", encoding="utf-8")
        analysis_data_5second = pd.read_csv(".//result//csv_result//" + str(video_name) + "//" + str(video_name) + "_smooth_5sec.csv", encoding="utf-8")
        #analysis_data_20second = pd.read_csv(".//result//csv_result//" + str(video_name) + "//" + str(video_name) + "_smooth_20sec.csv", encoding="shift_jis")
        #analysis_data_5second = pd.read_csv(".//result//csv_result//" + str(video_name) + "//" + str(video_name) + "_smooth_5sec.csv", encoding="shift_jis")

        #----------------------------------------------
        #データのグラフ化を行う(20secでの平滑化(鼻(R-B)))
        #----------------------------------------------

        if len(re_analysis_data_20second["Smooth R-B Value"]) != 0:

            re_max_graph = re_analysis_data_20second["Smooth R-B Value"].max() + 0.1
            re_min_graph = re_analysis_data_20second["Smooth R-B Value"].min() - 0.1
            max_graph = analysis_data_20second["Smooth R-B Value"].max() + 0.1
            min_graph = analysis_data_20second["Smooth R-B Value"].min() - 0.1

            max_time = analysis_data_20second["Time"].max() + 10

            #複数グラフを1つに表示するための準備
            fig = pyplot.figure(figsize=(15, 10))
            ax = fig.add_subplot(1, 1, 1)

            #平滑化したデータをグラフ化
            ax.plot("Time", "Smooth R-B Value", data=analysis_data_20second, color="red")
            ax.plot("Time", "Smooth R-B Value", data=re_analysis_data_20second, color="red", linestyle = "dashed")
            ax.set_ylim(min(min_graph, re_min_graph), max(max_graph, re_max_graph))

            sd = re_analysis_data_20second["Smooth R-B Value"].std(ddof=0)

            #グラフの諸設定(fig_nose_20sec_rb)
            pyplot.title("Nose R-B Value Analysis Result(20sec Smooth)", fontname="MS Gothic") #グラフタイトル
            pyplot.xlabel("Time(sec)", fontname="MS Gothic") #x軸
            pyplot.xticks(np.arange(0, max_time, 20), fontsize=5) #x軸のメモリを増加
            pyplot.ylabel("Value", fontname="MS Gothic") #y軸
            pyplot.minorticks_on() #補助線の追加
            pyplot.grid(axis="y") #y軸の目盛り線
            pyplot.legend(["All & Identification Data R-B Value",  "Standard Deviation:" + f'{sd:.3f}']) #凡例

            if number == 0:
                fig.savefig(".//result//figure_result//" + str(video_name) + "//" + str(interval) + "_" + str(step) + "//" + str(int(step) * int(number)) + "_" + str(int(interval) - 1) +"_20sec.png")
                pyplot.close(fig)
            else:
                fig.savefig(".//result//figure_result//" + str(video_name) + "//" + str(interval) + "_" + str(step) + "//" + str(int(step) * int(number)) + "_" + str(int(interval) - 1 + int(step) * int(number)) +"_20sec.png")
                pyplot.close(fig)

        #----------------------------------------------
        #データのグラフ化を行う(5secでの平滑化(鼻(R-B)))
        #----------------------------------------------

        if len(analysis_data_5second["Smooth R-B Value"]) != 0:

            re_max_graph_2 = re_analysis_data_5second["Smooth R-B Value"].max() + 0.1
            re_min_graph_2 = re_analysis_data_5second["Smooth R-B Value"].min() - 0.1
            max_graph_2 = analysis_data_5second["Smooth R-B Value"].max() + 0.1
            min_graph_2 = analysis_data_5second["Smooth R-B Value"].min() - 0.1

            max_time_2 = analysis_data_5second["Time"].max() + 10

            #複数グラフを1つに表示するための準備
            fig_2 = pyplot.figure(figsize=(15, 10))
            ax_2 = fig_2.add_subplot(1, 1, 1)

            #平滑化したデータをグラフ化
            ax_2.plot("Time", "Smooth R-B Value", data=analysis_data_5second, color="red")
            ax_2.plot("Time", "Smooth R-B Value", data=re_analysis_data_5second, color="red", linestyle = "dashed")
            ax_2.set_ylim(min(min_graph_2, re_min_graph_2), max(max_graph_2, re_max_graph_2))

            sd_2 = re_analysis_data_5second["Smooth R-B Value"].std(ddof=0)

            #グラフの諸設定(fig_nose_20sec_rb)
            pyplot.title("Nose R-B Value Analysis Result(5sec Smooth)", fontname="MS Gothic") #グラフタイトル
            pyplot.xlabel("Time(sec)", fontname="MS Gothic") #x軸
            pyplot.xticks(np.arange(0, max_time_2, 20), fontsize=5) #x軸のメモリを増加
            pyplot.ylabel("Value", fontname="MS Gothic") #y軸
            pyplot.minorticks_on() #補助線の追加
            pyplot.grid(axis="y") #y軸の目盛り線
            pyplot.legend(["All & Identification Data R-B Value",  "Standard Deviation:" + f'{sd:.3f}']) #凡例

            if number == 0:
                fig_2.savefig(".//result//figure_result//" + str(video_name) + "//" + str(interval) + "_" + str(step) + "//" + str(int(step) * int(number)) + "_" + str(int(interval) - 1) +"_5sec.png")
                pyplot.close(fig_2)
            else:
                fig_2.savefig(".//result//figure_result//" + str(video_name) + "//" + str(interval) + "_" + str(step) + "//" + str(int(step) * int(number)) + "_" + str(int(interval) - 1 + int(step) * int(number)) +"_5sec.png")
                pyplot.close(fig_2)



#-------------
#メイン関数
#-------------
if __name__=="__main__":

    interval = int(input("範囲の幅を数値入力してください："))
    step = int(input("ずらしていく幅を数値入力してください："))

    video_count = 0 #全ての動画に対して処理を行うためのカウント
    video_name_path = ".//stay_movie_path.txt" #ビデオを読み込みする

    with open(video_name_path) as f: #全動画を記したテキストを読み込む
        all_video = f.read().splitlines() #リストにする

        while True: #「True=動画が存在する」間回し続ける
            frame_count = 1 #CSVファイルの時間を書き込むためのカウント(初期設定)
            count = 1 #全フレーム(1秒に30枚)に対してランドマークはしないのでカウントフラグを使う(初期設定)

            try: #例外処理
                cap = cv2.VideoCapture(".//data_movie//" + str(all_video[video_count])) #動画を読み込む
            except IndexError: #読み込む動画の配列がなくなったら(リストを全て読み込んだら)
                break #whileから抜ける

            video_name = all_video[video_count].split(".")[0] #.mp4以前の名前を取得する

            #csvファイルを入れるフォルダを作成する
            csv_path = ".//result//csv_result//" + str(video_name) + "//" + str(interval) + "_" + str(step)
            os.mkdir(csv_path)
            #pngファイルを入れるフォルダを作成する
            figure_path = ".//result//figure_result//" + str(video_name) + "//" + str(interval) + "_" + str(step)
            os.mkdir(figure_path)

            file_number = ((765 - interval) // (step - 1)) + 2 #何個のファイルができるか計算する
            #RGB情報を書きこむ為のcsvファイルをあらかじめ全て作っておく
            for number in range(file_number):
                if number == 0:
                    f = open(".//result//csv_result//" + str(video_name) + "//" + str(interval) + "_" + str(step) + "//" + str(int(step) * int(number)) + "_" + str(int(interval) - 1) +"_landmark.csv", "a") #新規作成モードでファイルを開く
                    f.write("R Value" + "," + "G Value" + "," + "B Value" +  "," + "R-G Value" + "," + "R-B Value" + "," + "G-B Value" + "," + "Right R Value" + "," + "Right G Value" + "," + "Right B Value" +  "," + "Right R-G Value" + "," + "Right R-B Value" + "," + "Right G-B Value" + "," + "Left R Value" + "," + "Left G Value" + "," + "Left B Value" +  "," + "Left R-G Value" + "," + "Left R-B Value" + "," + "Left G-B Value" + "," + "Both R Value" + "," + "Both G Value" + "," + "Both B Value" +  "," + "Both R-G Value" + "," + "Both R-B Value" + "," + "Both G-B Value" + "," + "Time" + "\n") #ヘッダー作成
                    f.close()
                else:
                    f = open(".//result//csv_result//" + str(video_name) + "//" + str(interval) + "_" + str(step) + "//" + str(int(step) * int(number)) + "_" + str(int(interval) - 1 + int(step) * int(number)) +"_landmark.csv", "a") #新規作成モードでファイルを開く
                    f.write("R Value" + "," + "G Value" + "," + "B Value" +  "," + "R-G Value" + "," + "R-B Value" + "," + "G-B Value" + "," + "Right R Value" + "," + "Right G Value" + "," + "Right B Value" +  "," + "Right R-G Value" + "," + "Right R-B Value" + "," + "Right G-B Value" + "," + "Left R Value" + "," + "Left G Value" + "," + "Left B Value" +  "," + "Left R-G Value" + "," + "Left R-B Value" + "," + "Left G-B Value" + "," + "Both R Value" + "," + "Both G Value" + "," + "Both B Value" +  "," + "Both R-G Value" + "," + "Both R-B Value" + "," + "Both G-B Value" + "," + "Time" + "\n") #ヘッダー作成
                    f.close()

            while True: #動画が終わるまで続ける
                ret, img = cap.read()

                if ret == False: #もしretがFalseだったら
                    break #動画の画像は1つ前でなくなっているのでループから抜ける

                if count == 30: #1秒経過していたら
                    count = 1 #カウンターを初期化
                    face_landmark_facemesh(img, frame_count, video_name, interval, step) #取り出したimgに対してランドマーク
                    frame_count = frame_count + 1 #1増やす

                else: #countが30未満だったら
                    count = count + 1 #countを増やす

            cap.release() #動画を終了する
            video_count += 1 #次の動画を見るためにカウンターを増やす

            smooth_data(interval, step) #平滑化したデータを作成する
            make_graph(interval, step) #時系列グラフの作成