1、在原来的labelImg上进行的二次开发
2、增加了文件校验通过、校验错误、文件删除、自动保存、是否遮挡等信息的更新
3、图中标签的展示，可以在最上头的view文件夹中打开，根据需要，我调整了显示的透明度
4、预设classes ,但是无法封装需要在运行文件路径的./data/predefined_classes.txt中设置

5、增加了一个视频类数据自动标注按钮(根据前一张标注和追踪模型自动标注)







经过测试，仍然存在的bug:
1、在Ubuntu下最大化窗口会直接关闭(ubuntu系统服务器不会，但是使用xshell会)
2、Ubuntu下create bbox,只有第一次是弹出预设预设classes选择的，
   其他都是默认第一个并且编辑修改只能手动退出绘框模式
3、绘图模式下，操作右侧栏按钮信息无效，必须退出绘图模型才行
4、选择了一个单选类（eg.person）,绘制一个person 后立马全部框都会重现



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


软件封装：(首先使用pip安装pyinstaller)
pyinstaller -F labelImg.py


项目文件简易说明：
项目主程序文件

labelImg.py ：可增减按钮等信息（增加信息需要修改resources/strings下的string.properties文字信息，icons下放置图标。 修改需要重新编译）

图片路径文件在resources.qrc中

canvas.py 中间显示框的鼠标事件等信息

shape.py 用来设置显示颜色和透明度等信息

stringBundle.py 主要是读取配置文本信息

labelFile.py 修改的存储xml的相关信息

如果修改涉及到输出的xml标注文件信息的需要修改 labelImg.py/pascal_voc_io.py/yolo_io.py/labelFile.py