# import numpy as np
# import matplotlib.pyplot as plt

# # 创建一个2D meshgrid
# x = np.linspace(-3.0, 3.0, 100)
# y = np.linspace(-3.0, 3.0, 100)
# X, Y = np.meshgrid(x, y)

# # 计算每个点的unsigned distance value
# # 这里我们使用一个简单的函数：Z = X**2 + Y**2
# Z = X**2 + Y**2

# # 设置iso-value
# v = 2.5

# # 调用marching_squares函数
# contours = marching_squares(Z, v)

# # 绘制结果
# for contour in contours:
#     plt.plot(contour[:, 0], contour[:, 1], 'k-')

# plt.show()


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
import cv2
from PIL import Image, ImageDraw

class Square():
    A = [0, 0]
    B = [0, 0]
    C = [0, 0]
    D = [0, 0]
    A_data = 0.0
    B_data = 0.0
    C_data = 0.0
    D_data = 0.0

    def GetCaseId(self, threshold):
        caseId = 0
        if (self.A_data >= threshold):
            caseId |= 1
        if (self.B_data >= threshold):
            caseId |= 2
        if (self.C_data >= threshold):
            caseId |= 4
        if (self.D_data >= threshold):
            caseId |= 8
            
        return caseId

    def GetLines(self, Threshold):
        lines = []
        caseId = self.GetCaseId(Threshold)

        if caseId in (0, 15):
            return []

        if caseId in (1, 14, 10):
            pX = (self.A[0] + self.B[0]) / 2
            pY = self.B[1]
            qX = self.D[0]
            qY = (self.A[1] + self.D[1]) / 2

            line = (pX, pY, qX, qY)

            lines.append(line)

        if caseId in (2, 13, 5):
            pX = (self.A[0] + self.B[0]) / 2
            pY = self.A[1]
            qX = self.C[0]
            qY = (self.A[1] + self.D[1]) / 2

            line = (pX, pY, qX, qY)

            lines.append(line)

        if caseId in (3, 12):
            pX = self.A[0]
            pY = (self.A[1] + self.D[1]) / 2
            qX = self.C[0]
            qY = (self.B[1] + self.C[1]) / 2

            line = (pX, pY, qX, qY)

            lines.append(line)

        if caseId in (4, 11, 10):
            pX = (self.C[0] + self.D[0]) / 2
            pY = self.D[1]
            qX = self.B[0]
            qY = (self.B[1] + self.C[1]) / 2

            line = (pX, pY, qX, qY)

            lines.append(line)

        elif caseId in (6, 9):
            pX = (self.A[0] + self.B[0]) / 2
            pY = self.A[1]
            qX = (self.C[0] + self.D[0]) / 2
            qY = self.C[1]

            line = (pX, pY, qX, qY)

            lines.append(line)

        elif caseId in (7, 8, 5):
            pX = (self.C[0] + self.D[0]) / 2
            pY = self.C[1]
            qX = self.A[0]
            qY = (self.A[1] + self.D[1]) / 2

            line = (pX, pY, qX, qY)

            lines.append(line)

        return lines

def marching_square(xVector, yVector, Data, threshold):
    linesList = []

    Height = len(Data)  # rows
    Width = len(Data[1])  # cols

    if ((Width == len(xVector)) and (Height == len(yVector))):
        squares = np.full((Height - 1, Width - 1), Square())

        sqHeight = squares.shape[0]  # rows count
        sqWidth = squares.shape[1]  # cols count

        for j in range(sqHeight):  # rows
            for i in range(sqWidth):  # cols
                a = Data[j + 1, i]
                b = Data[j + 1, i + 1]
                c = Data[j, i + 1]
                d = Data[j, i]
                A = [xVector[i], yVector[j + 1]]
                B = [xVector[i + 1], yVector[j + 1]]
                C = [xVector[i + 1], yVector[j]]
                D = [xVector[i], yVector[j]]

                squares[j, i].A_data = a
                squares[j, i].B_data = b
                squares[j, i].C_data = c
                squares[j, i].D_data = d

                squares[j, i].A = A
                squares[j, i].B = B
                squares[j, i].C = C
                squares[j, i].D = D

                list = squares[j, i].GetLines(threshold)

                linesList = linesList + list
    else:
        raise AssertionError

    return [linesList]

# ------------------------------------------------------------------------------ #
def re_arrange_order(contour):
    contour_new = []
    idx = 0
    cont_idx = contour[idx]
    contour_new.append(cont_idx)
    contour = np.delete(contour, idx, axis=0)
    while contour.any():
        assert (cont_idx == contour_new[-1]).all()
        cont_i_distance = np.linalg.norm((contour - cont_idx), axis=1)
        idx = np.argmin(cont_i_distance)
        cont_idx = contour[idx]
        contour_new.append(cont_idx)
        contour = np.delete(contour, idx, axis=0)
    contour_new.append(contour_new[0])
    return np.array(contour_new)
def contours_process(contours):
    ret = []
    for contour in contours:
        ret_iter = []
        for c in contour:
            ret_iter.append([c[0], c[1]])
            ret_iter.append([c[2], c[3]])
        ret_iter = re_arrange_order(np.array(ret_iter))
        ret.append(ret_iter)
    return ret
# ------------------------------------------------------------------------------ #

def main():
    x = [i for i in range(256)]
    y = [i for i in range(256)]
    
    example_l = [[0 for i in range(256)] for j in range(256)]
    
    for i in range(len(example_l)):
        for j in range(len(example_l[0])):
            example_l[i][j] = math.sin(i / 10.0)*math.cos(j / 10.0)
    example = np.array(example_l)
    example = abs(example)

    import ipdb; ipdb.set_trace()

    im = Image.new('RGB', (256, 256), (128, 128, 128))

    collection = marching_square(x, y, example, 0.9)

    draw = ImageDraw.Draw(im)
    xx = []
    yy = []
    for ln in collection:
        for toup in ln:
            xx.append(toup[0])
            xx.append(toup[2])

            yy.append(toup[1])
            yy.append(toup[3])
            # draw.line(toup, fill=(255, 255, 0), width=1)
        
    import ipdb; ipdb.set_trace()
    plt.scatter(xx, yy)
    plt.savefig("aa.jpg")
    # im.save("aa.jpg")


if __name__ == '__main__':
    main()