1、在原来的labelImg上进行的二次开发
2、增加了文件校验通过、校验错误、文件删除、自动保存、是否遮挡等信息的更新
3、图中标签的展示，可以在最上头的view文件夹中打开，根据需要，我调整了显示的透明度
4、预设classes ,但是无法封装需要在运行文件路径的./data/predefined_classes.txt中设置

经过测试，仍然存在的bug:
1、在Ubuntu下最大化窗口会直接关闭
2、Ubuntu下create bbox,只有第一次是弹出预设预设classes选择的，
   其他都是默认第一个并且编辑修改只能手动退出绘框模式



编译：：
Ubuntu Linux

Python 3 + Qt5 (Recommended)

    sudo apt-get install pyqt5-dev-tools
    sudo pip3 install -r requirements/requirements-linux-python3.txt
    make qt5py3
    python3 labelImg.py
    python3 labelImg.py [IMAGE_PATH] [PRE-DEFINED CLASS FILE]


Windows + Anaconda

    conda install pyqt=5
    conda install -c anaconda lxml
    pyrcc5 -o libs/resources.py resources.qrc
    python labelImg.py
    python labelImg.py [IMAGE_PATH] [PRE-DEFINED CLASS FILE]




项目文件简易说明：
项目主程序文件

labelImg.py ：可增减按钮等信息（增加信息需要修改resources/strings下的string.properties文字信息，icons下放置图标。 修改需要重新编译）


canvas.py 中间显示框的鼠标事件等信息

shape.py 用来设置显示颜色和透明度等信息

stringBundle.py 主要是读取配置文本信息

labelFile.py 修改的存储xml的相关信息

如果修改涉及到输出的xml标注文件信息的需要修改 labelImg.py/pascal_voc_io.py/yolo_io.py/labelFile.py