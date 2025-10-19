import matplotlib.pyplot as plt

from pilotscope.Common.Util import sum_list


class Drawer:

    @classmethod
    def draw_bar(cls, name_2_values: dict, file_name, x_title="algos", y_title="Time(s)", is_rotation=False):
        names = list(name_2_values.keys())

        values = []
        for vs in name_2_values.values():
            values.append(sum_list(vs) / len(vs) if hasattr(vs, '__iter__') else vs)

        plt.figure(figsize=(6, 4))
        plt.bar(names, values, width=0.4)
        # x축 범위를 데이터 개수에 맞게 동적 설정
        plt.xlim(-0.5, len(names) - 0.5)

        # 设置x轴和y轴标签
        plt.xlabel(x_title)
        plt.ylabel(y_title)
        if is_rotation:
            plt.xticks(rotation=45)
        plt.subplots_adjust(bottom=0.2)
        # 保存图像
        plt.savefig(file_name+'.png')
        plt.show()

    @classmethod
    def draw_line(cls, name_2_values: dict):
        pass
