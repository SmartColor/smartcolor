<template>
  <div class="explore-box">
<!--    <h1>Explore</h1>-->
    <div class="search">
      <el-input
        placeholder="使用颜色、情境、关键词等进行搜索，例如：橙色、快乐..."
        v-model="input"
        clearable
      >
        <el-button slot="append" icon="el-icon-search" @click="search()"></el-button>
      </el-input>
    </div>

    <div class="container">
      <div id="sub-header">
        👋 Click the color block to get the color value !
      </div>
      <div class="color-con">
        <!--<el-row type="flex" class="row-bg" justify="space-between" v-for='(item,index) in colors' :key="index">

          <el-col :span="7" v-for='(item1,index1) in item' :key="index1">
            <div class="grid-content">
              <ul class="color-box">
                <li v-for='(items,indexs) in item1' :key="indexs" :style="{backgroundColor:
              'rgb('+items[0]+','+items[1]+','+items[2]+')'}"></li>
              </ul>
            </div>
          </el-col>

        </el-row>-->
        <textarea id="input" style="display: none;">这是幕后黑手</textarea>
        <ul class="color-box">
          <li v-for='(item,index) in colors' :key="index">
          <!--  <div
              class="viewer"><span>100</span><i
              class="el-icon-view"></i><span>10</span><i class="el-icon-star-off"></i></div>-->
            <ul class="small-box">
              <li v-for='(item1,index1) in item' :key="index1" :style="{backgroundColor:
              'rgb('+item1[0]+','+item1[1]+','+item1[2]+')'}" @click="copyText(item1)"></li>
            </ul>
          </li>
        </ul>
      </div>

      <div class="pagination-box">
        <el-pagination
          background
          layout="prev,next"
          :total="50">
        </el-pagination>
      </div>
    </div>

  </div>
</template>

<script>
export default {
  name: 'Explore',
  data () {
    return {
      input: '',
      colors: [
        [[82, 232, 246], [13, 154, 255], [4, 138, 129], [222, 156, 154], [133, 108, 107]],
        [[33, 175, 112], [15, 89, 56], [255, 255, 255], [0, 0, 0], [16, 69, 71]],
        [[13, 154, 255], [82, 232, 246], [222, 156, 154], [237, 203, 150], [133, 108, 107]],
        [[13, 154, 255], [82, 232, 246], [222, 156, 154], [233, 217, 133], [133, 108, 107]],
        [[34, 34, 59], [22, 159, 191], [142, 225, 239], [241, 143, 1], [255, 255, 255]],
        [[253, 117, 0], [0, 133, 202], [241, 242, 235], [252, 252, 252], [41, 47, 54]],
        [[13, 154, 255], [82, 232, 246], [75, 255, 239], [54, 234, 170], [60, 254, 130]],
        [[22, 159, 191], [142, 225, 239], [241, 143, 1], [190, 233, 232], [255, 255, 255]],
        [[246, 96, 133], [237, 105, 50], [242, 147, 14], [237, 195, 30], [46, 204, 113]],
        [[178, 76, 125], [52, 169, 237], [109, 118, 196], [85, 244, 210], [75, 36, 173]],
        [[5, 5, 5], [255, 16, 83], [108, 110, 160], [245, 241, 227], [255, 255, 255]],
        [[37, 40, 61], [22, 159, 191], [142, 225, 239], [211, 32, 84], [255, 255, 255]],
        [[128, 141, 160], [149, 169, 179], [255, 245, 245], [207, 180, 188], [247, 202, 183]],
        [[51, 102, 153], [22, 159, 191], [142, 225, 239], [211, 32, 84], [255, 255, 255]],
        [[218, 215, 205], [215, 138, 118], [247, 179, 43], [12, 131, 70], [0, 79, 45]],
        [[229, 218, 208], [190, 191, 176], [94, 11, 21], [144, 50, 61], [188, 128, 52]],
        [[136, 54, 119], [61, 90, 128], [96, 96, 98], [242, 197, 124], [237, 50, 55]],
        [[241, 237, 234], [192, 184, 156], [154, 74, 51], [246, 56, 35], [42, 16, 8]],
        [[200, 29, 37], [120, 17, 26], [128, 26, 25], [0, 0, 0], [40, 40, 25]],
        [[250, 153, 115], [251, 177, 148], [253, 217, 203], [255, 232, 223], [45, 30, 47]],
        [[43, 45, 66], [141, 153, 174], [237, 242, 244], [238, 193, 112], [217, 4, 41]],
        [[221, 215, 141], [114, 14, 7], [220, 191, 133], [139, 99, 92], [190, 213, 88]],
        [[182, 208, 148], [114, 14, 7], [225, 170, 125], [190, 138, 96], [190, 213, 88]],
        [[89, 0, 44], [248, 244, 227], [208, 196, 223], [206, 231, 230], [191, 209, 229]],
        [[243, 216, 113], [254, 144, 135], [37, 14, 4], [235, 79, 152], [205, 173, 72]],
        [[32, 39, 40], [255, 251, 252], [147, 158, 49], [98, 187, 193], [216, 190, 123]],
        [[90, 202, 218], [246, 232, 211], [153, 229, 244], [180, 197, 105], [253, 213, 65]],
        [[206, 208, 161], [112, 214, 255], [255, 112, 166], [255, 151, 112], [247, 179, 43]]
      ]
    }
  },
  methods: {
    copyText: function (color) {
      // var text = document.getElementById("text").innerText;
      // var text = document.getElementById('text').attributes['color'].nodeValue
      var input = document.getElementById('input')
      input.value = color // 修改文本框的内容
      input.select() // 选中文本
      document.execCommand('copy') // 执行浏览器复制命令
      this.$notify({
        title: '成功',
        message: color + ' 复制成功!',
        type: 'success'
      })
      // alert(color + '复制成功')
    },
    search: function () {
      this.colors = [
        [[234, 84, 20], [157, 52, 7], [255, 109, 47], [0, 157, 136], [20, 234, 205]],
        [[104, 94, 96], [243, 165, 101], [213, 108, 36], [216, 170, 154], [231, 198, 194]],
        [[223, 173, 0], [255, 135, 76], [255, 115, 47], [0, 178, 163], [155, 247, 239]],
        [[223, 73, 0], [89, 85, 27], [242, 182, 128], [242, 143, 121], [13, 0, 0]],
        [[1, 160, 200], [250, 254, 253], [255, 110, 1], [245, 91, 5], [242, 242, 244]],
        [[202, 112, 53], [211, 138, 89], [226, 178, 146], [238, 210, 191], [252, 246, 242]],
        [[1, 204, 1], [54, 207, 239], [244, 175, 54], [243, 91, 5], [255, 255, 255]],
        [[2, 121, 163], [2, 83, 117], [1, 134, 167], [241, 226, 2], [241, 146, 90]],
        [[242, 248, 92], [90, 90, 92], [242, 183, 10], [241, 85, 22], [244, 244, 242]],
        [[255, 71, 21], [251, 176, 107], [255, 177, 86], [254, 112, 36], [253, 76, 39]],
        [[192, 91, 65], [52, 47, 44], [184, 80, 54], [176, 161, 136], [202, 75, 61]],
        [[15, 9, 38], [242, 174, 48], [217, 144, 54], [217, 88, 59], [166, 40, 28]],
        [[255, 175, 12], [232, 136, 11], [255, 107, 1], [232, 71, 11], [255, 42, 12]],
        [[228, 225, 206], [179, 209, 220], [99, 141, 31], [254, 150, 17], [240, 76, 13]],
        [[153, 171, 187], [31, 88, 161], [231, 142, 87], [168, 75, 14], [251, 140, 74]],
        [[156, 220, 77], [246, 221, 98], [247, 118, 36], [244, 244, 244], [141, 141, 141]],
        [[127, 31, 0], [255, 121, 77], [255, 62, 0], [127, 60, 39], [204, 50, 0]],
        [[255, 208, 13], [232, 167, 12], [255, 156, 0], [232, 114, 12], [255, 90, 13]],
        [[242, 209, 148], [242, 185, 80], [242, 139, 48], [217, 72, 20], [166, 52, 27]],
        [[0, 0, 0], [255, 236, 212], [255, 159, 31], [0, 115, 179], [31, 175, 255]],
        [[45, 109, 166], [206, 242, 128], [242, 182, 109], [217, 149, 89], [191, 148, 132]],
        [[255, 84, 13], [255, 49, 14], [255, 0, 0], [232, 0, 116], [161, 8, 161]],
        [[165, 255, 23], [222, 185, 98], [252, 69, 35], [185, 156, 196], [39, 118, 249]],
        [[219, 199, 172], [197, 175, 152], [233, 180, 40], [176, 113, 69], [170, 29, 38]],
        [[254, 127, 23], [229, 83, 20], [252, 69, 35], [229, 22, 20], [254, 23, 130]],
        [[46, 86, 166], [162, 197, 242], [242, 242, 242], [172, 222, 242], [242, 239, 48]],
        [[84, 64, 39], [217, 192, 145], [242, 224, 208], [187, 173, 111], [242, 120, 75]]
        // [[206, 208, 161], [112, 214, 255], [255, 112, 166], [255, 151, 112], [247, 179, 43]]
      ]
    }
  }
}
// 使用颜色、情境、关键词等进行搜索，例如：黄色、海洋
</script>

<style scoped>
  .el-col {
    border-radius: 4px;
  }
  .bg-purple-dark {
    background: #99a9bf;
  }
  .bg-purple {
    background: #d3dce6;
  }
  .bg-purple-light {
    background: #e5e9f2;
  }
  .grid-content {
    border-radius: 4px;
    min-height: 36px;
  }

  #sub-header{
    /*background-color: #FFFFFF;*/
    height: 60px;
    line-height: 60px;
    text-align: left;
    color: #333;
    padding: 0 80px;
    font-family: Inter UI,-apple-system,BlinkMacSystemFont,Segoe UI,Roboto,Oxygen,Ubuntu,Cantarell,Fira Sans,Droid Sans,Helvetica Neue,sans-serif;
    /*font-size: 18px;*/
  }

  .search{
    padding: 20px 80px 20px 80px ;
  }

  .el-input--suffix .el-input__inner {
    padding-right: 60px;
  }

  .explore-box{
    background-color: #ffffff;
  }
  .container{
    background-color: #F9F9F9;
  }

  .color-con{
    padding: 0 80px;
    overflow: hidden;
  }
  .pagination-box{
    /*margin-top: 20px;*/
    padding-bottom: 15px;
  }

  /************************************************/
  .color-box{
    width: 100%;
    height: 84px;
    list-style: none;
    /*display: flex;
    justify-content: space-between;*/
  }
  .color-box>li{
    width: 13%;
    height: 100%;
    float: left;
    /*background-color: yellow;*/
    margin-right: 1.5%;
    margin-bottom: 26px;
    border-radius: 4px;
    overflow: hidden;
  }

  .color-box>li:nth-child(7n){
    margin-right: 0;
  }
  .small-box{
    width: 100%;
    height:100%;
    border-radius: 4px;
    overflow: hidden;
  }
  .small-box>li{
    width: 20%;
    height: 100%;
    float: left;
    z-index: 222;
  }

  .viewer{
    width: 100%;
    height: 30px;
    /*background-color: #969896;*/
    line-height: 30px;
    color: #c3c3c3;
    font-size: 12px;
  }
  .viewer>i{
    float: right;
    margin-right: 10px;
    line-height: 30px;
  }
  .viewer>span{
    float: right;
    margin-right: 10px;
  }
  /****************************************************/
</style>
