<template>
  <div>
    <div class="website">

      <template>
        <div id="sub-header">
          👋 Click the color block to get the color value !
          <!--<el-button icon="el-icon-search" style="float: right;" size="medium" round>Search</el-button>-->
          <i class="el-icon-search" style="float: right;margin-right: 15%;margin-left: 11px;line-height: 50px"
             @click="change()"></i>

          <div class="block inlineBlock">
            <!--<span class="demonstration">默认</span>-->
            <el-date-picker
              v-model="value1"
              type="daterange"
              range-separator="至"
              start-placeholder="START"
              end-placeholder="END">
            </el-date-picker>
          </div>

        </div>

      </template>

      <!--    <p>Click the color block to get the color value!</p>-->
      <div style="overflow: hidden">
        <textarea id="input" style="display: none;">这是幕后黑手</textarea>
       <!-- <el-row type="flex" class="row-bg" justify="space-around" v-for='(item,index) in colors' :key="index">

          <el-col :span="7" v-for='(item1,index1) in item' :key="index1" >
              <div class="grid-content">
                <ul class="color-box">
                  <li v-for='(items,indexs) in item1' :key="indexs" :style="{backgroundColor:items}"
                      @click="copyText(items)"></li>
                </ul>
              </div>
          </el-col>

         &lt;!&ndash; <el-col :span="7">
            <div class="grid-content">
              <ul class="color-box">
                <li style="background-color: yellow"></li>
                <li style="background-color: pink"></li>
                <li style="background-color: yellow"></li>
                <li style="background-color: pink"></li>
                <li style="background-color: blue"></li>
              </ul>
            </div>
          </el-col>&ndash;&gt;

        </el-row>-->
        <div class="mask" v-if="showModal" @click="showModal=false">
          <ul>
            <li style="backgroundColor:rgb(82, 232, 246)">#52E8F6</li>
            <li style="backgroundColor:rgb(13, 154, 255)">#0D9AFF</li>
            <li style="backgroundColor:rgb(4, 138, 129)">#048A81</li>
            <li style="backgroundColor:rgb(222, 156, 154)">#DE9C9A</li>
            <li style="backgroundColor:rgb(133, 108, 107)">#856C6B</li>
          </ul>
        </div>
        <ul class="color-box">

          <li>
            <div
              class="viewer"><span>{{viewerCount}}</span><i
              class="el-icon-view" @click="showModal=true;viewerAdd()"></i><span>{{collectCount}}</span><i
              class="el-icon-star-off"
                                                                       @click="collectAdd()"></i>
            </div>
            <ul class="small-box">
              <li style="backgroundColor:rgb(82, 232, 246)"></li>
              <li style="backgroundColor:rgb(13, 154, 255)" ></li>
              <li style="backgroundColor:rgb(4, 138, 129)"></li>
              <li style="backgroundColor:rgb(222, 156, 154)"></li>
              <li style="backgroundColor:rgb(133, 108, 107)"></li>
              <!--[[82, 232, 246], [13, 154, 255], [4, 138, 129], [222, 156, 154], [133, 108, 107]],-->
            </ul>
          </li>
          <!--*************-->
          <li v-for='(item,index) in colors' :key="index">
            <div
              class="viewer"><span>100</span><i
              class="el-icon-view"></i><span>10</span><i class="el-icon-star-off"></i></div>
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
    <div class="advantage">
       <div class="advantageCon">
         <h4 class="advantageTitle">Our Advantages</h4>
          <ul>
            <li>
                <div class="subTitle">
                <div class="subPicture">
                  <img src="../../../static/image/menu.png" alt="">
                </div>
                  <div class="subContent">
                    <h4>Color Palettes</h4>
                    <p>ColorBlend is a 100% data-driven collection of color palettes. After a year of curating
                      beautiful designs over at Pixels I figured out that I could make something cool with the data and this is the result.</p>
                  </div>
              </div>
            </li>
            <li>
              <div class="subTitle">
                <div class="subPicture">
                  <img src="../../../static/image/explore.png" alt="">
                </div>
                <div class="subContent">
                  <h4>Explore The Color</h4>
                  <p>According to people's emotional psychology and the use of situational intelligence to generate
                    some color schemes, the resulting color schemes echo with people's heart situation at that time, truly emphasizing people's subjective consciousness.</p>
                </div>
              </div>
            </li>
            <li>
              <div class="subTitle">
                <div class="subPicture">
                  <img src="../../../static/image/Generate.png" alt="">
                </div>
                <div class="subContent">
                  <h4>Generate Colors</h4>
                  <p>When we need to want more color matching, according to the large amount of color matching data
                    we generate, by locking one or two, three, four colors to generate the rest of the color, to provide users with a variety of color choices.</p>
                </div>
              </div>
            </li>
<!--当我们需想要更多的配色时，根据我们生成的大量配色数据，通过锁住其中一个或者两个、三个、四个颜色来生成其余的颜色，为用户提供多种配色选择-->
          </ul>
       </div>
    </div>
  </div>
</template>

<script>
export default {
  name: 'Website',
  data () {
    return {
      /* colors: [
        [['#171421', '#d3d0a3', '#85684c', '#37171f', '#4c4944'],
          ['#6e4a32', '#9698a2', '#a59d9a', '#434a46', '#a38963'],
          ['#debda9', '#ad776e', '#1e212b', '#3c3b37', '#cebec0']
        ],
        [
          ['#8eb158', '#7c8042', '#747769', '#53366e', '#c64b1f'],
          ['#291211', '#171923', '#57300e', '#483a32', '#451f0a'],
          ['#221d1d', '#a28057', '#40231b', '#513e32', '#75481e']
        ],
        [
          ['#232303', '#3bd7de', '#94e2ff', '#9bb3ea', '#d6a0f3'],
          ['#011e4d', '#060d21', '#01588b', '#736b87', '#91bfe1'],
          ['#a90000', '#466025', '#7a0104', '#1d2319', '#6d9632']
        ],
        [
          ['#275ea7', '#263d95', '#797994', '#a5c8e6', '#b88b6a'],
          ['#3d9aef', '#eda539', '#7a3f17', '#cc671d', '#7e9acc'],
          ['#b61d08', '#02a5d8', '#3a738e', '#152815', '#a9e9ea']
        ],
        [
          ['#054493', '#5b296c', '#93c3ef', '#4b7cbd', '#152958'],
          ['#1894d2', '#028802', '#017cd4', '#55e402', '#c40f19'],
          ['#7a09ff', '#c4c8d8', '#490ac4', '#73c4f1', '#311080']
        ],
        [
          ['#5694ef', '#8cc3ff', '#addfff', '#f3ac15', '#2668ac'],
          ['#0b1d79', '#061144', '#132ebd', '#060d25', '#fdff15'],
          ['#252daa', '#454541', '#ecbc79', '#4951f4', '#a88a12']
        ],
        [
          ['#0183dd', '#d74341', '#1894d2', '#014a97', '#fed3a2'],
          ['#573c32', '#98694d', '#d09160', '#f4e7cd', '#2d437c'],
          ['#cbd4e9', '#dca30e', '#94bce6', '#6f8423', '#79e097']
        ]

      ] */
      value1: '',
      value2: '',
      viewerCount: 100,
      collectCount: 10,
      showModal: false,
      colors: [
        // [[82, 232, 246], [13, 154, 255], [4, 138, 129], [222, 156, 154], [133, 108, 107]],
        [[206, 208, 161], [112, 214, 255], [255, 112, 166], [255, 151, 112], [247, 179, 43]],
        // [[33, 175, 112], [15, 89, 56], [255, 255, 255], [0, 0, 0], [16, 69, 71]],
        // [[13, 154, 255], [82, 232, 246], [222, 156, 154], [237, 203, 150], [133, 108, 107]],
        [[13, 154, 255], [82, 232, 246], [222, 156, 154], [233, 217, 133], [133, 108, 107]],
        // [[34, 34, 59], [22, 159, 191], [142, 225, 239], [241, 143, 1], [255, 255, 255]],
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
        // [[200, 29, 37], [120, 17, 26], [128, 26, 25], [0, 0, 0], [40, 40, 25]],
        [[250, 153, 115], [251, 177, 148], [253, 217, 203], [255, 232, 223], [45, 30, 47]],
        [[43, 45, 66], [141, 153, 174], [237, 242, 244], [238, 193, 112], [217, 4, 41]],
        [[221, 215, 141], [114, 14, 7], [220, 191, 133], [139, 99, 92], [190, 213, 88]],
        [[182, 208, 148], [114, 14, 7], [225, 170, 125], [190, 138, 96], [190, 213, 88]],
        [[89, 0, 44], [248, 244, 227], [208, 196, 223], [206, 231, 230], [191, 209, 229]],
        [[243, 216, 113], [254, 144, 135], [37, 14, 4], [235, 79, 152], [205, 173, 72]],
        [[32, 39, 40], [255, 251, 252], [147, 158, 49], [98, 187, 193], [216, 190, 123]],
        [[90, 202, 218], [246, 232, 211], [153, 229, 244], [180, 197, 105], [253, 213, 65]]

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
    collectAdd: function () {
      this.collectCount = this.collectCount + 1
    },
    viewerAdd: function () {
      this.viewerCount = this.viewerCount + 1
    },
    change () {
      this.colors = [
        [[45, 48, 71], [65, 157, 120], [255, 253, 130], [22, 244, 208], [255, 155, 113]],
        [[142, 74, 31], [224, 60, 221], [0, 0, 0], [227, 181, 164], [204, 251, 254]],
        [[46, 134, 171], [162, 59, 114], [228, 0, 102], [0, 175, 196], [206, 255, 0]],
        [[255, 234, 208], [247, 111, 142], [40, 43, 40], [255, 51, 31], [98, 131, 149]],
        [[130, 2, 99], [234, 222, 218], [228, 0, 102], [0, 175, 196], [206, 255, 0]],
        [[184, 12, 9], [11, 79, 108], [251, 251, 255], [0, 175, 196], [206, 255, 0]],
        [[1, 42, 69], [141, 2, 31], [217, 197, 151], [250, 251, 237], [255, 255, 255]],
        [[61, 8, 20], [109, 33, 60], [23, 55, 83], [87, 92, 85], [127, 123, 130]],
        [[224, 85, 48], [231, 137, 53], [244, 201, 199], [138, 172, 143], [129, 200, 213]],
        [[45, 48, 71], [240, 240, 201], [255, 201, 181], [247, 177, 171], [255, 155, 113]],
        [[99, 71, 77], [166, 61, 64], [233, 184, 114], [163, 119, 116], [194, 185, 127]],
        [[247, 60, 49], [249, 153, 46], [249, 176, 175], [138, 172, 143], [49, 92, 73]],
        [[1, 57, 94], [139, 12, 57], [255, 255, 255], [250, 251, 237], [217, 197, 151]],
        [[223, 154, 87], [52, 64, 85], [51, 30, 54], [31, 39, 27], [51, 44, 35]],
        [[242, 208, 211], [249, 248, 242], [229, 205, 190], [244, 219, 216], [251, 233, 232]],
        [[123, 45, 38], [153, 70, 54], [64, 61, 88], [98, 131, 149], [177, 182, 149]],
        [[255, 255, 255], [236, 102, 79], [232, 129, 61], [235, 232, 96], [160, 196, 91]],
        [[87, 95, 131], [239, 232, 230], [98, 159, 213], [222, 158, 130], [139, 174, 200]],
        [[124, 93, 110], [245, 211, 208], [239, 199, 194], [249, 221, 219], [212, 193, 186]],
        [[190, 82, 70], [98, 159, 213], [182, 182, 209], [114, 89, 79], [191, 152, 130]],
        [[98, 159, 213], [191, 152, 130], [127, 190, 234], [206, 212, 224], [212, 201, 192]],
        [[68, 55, 66], [109, 89, 75], [206, 160, 126], [237, 217, 163], [219, 228, 168]]

      ]
    }

  }
}
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

  .website{
    background-color: #191919;
    padding: 30px 0px 60px 0;
  }

  #sub-header{
    /*background-color: #FFFFFF;*/
    height: 50px;
    line-height: 50px;
    text-align: left;
    color: #c3c3c3;
    padding: 0 42px;
    font-family: Inter UI,-apple-system,BlinkMacSystemFont,Segoe UI,Roboto,Oxygen,Ubuntu,Cantarell,Fira Sans,Droid Sans,Helvetica Neue,sans-serif;
    /*font-size: 18px;*/
  }

 /* .color-box{
    width: 100%;
    height: 51px;
    list-style: none;
    border-radius: 4px;
    overflow: hidden;
  }
  .color-box li{
    width: 20%;
    height: 100%;
    float: left;
  }
*/
  .pagination-box{
    margin-top: 30px;
  }
  .el-pagination.is-background .el-pager li:not(.disabled).active{
    background-color: rgb(255, 208, 75)!important;
    color: #FFF;
  }

  .advantage{
    width: 100%;
    height: 580px;
    background-color: #F4F5EF;
    padding-top: 35px;
  }
  .advantageCon{
    width: 60%;
    height: 100%;
    /*background-color: #5daf34;*/
    margin: 0 auto;
    /*padding: 50px 0;*/
    box-sizing: border-box;
  }
  .advantageTitle{
    font-size: 1.825em;
    line-height: 3.4em;
    margin: 20px 0 10px;
    font-family: "Roboto", "Helvetica", "Arial", sans-serif;
    font-weight: 300;
    text-align: left;
  }
  .advantageCon>h4{
    margin: 0;
  }
  .advantageCon>ul{
    width: 100%;
    /*height: calc(100% - 3.4em);*/
    /*background-color: blue;*/
    display: flex;
    justify-content: space-between;
    list-style: none;
    margin: 0;
  }
  .advantageCon>ul>li{
    width: 33%;
    /*height: 100%;*/
    /*background-color: pink;*/
  }
  .subTitle{
    position: relative;
  }
  .subContent{
    text-align: left;
    margin-left: 89px;
    text-align: justify;
    padding-right: 40px;
  }
  .subPicture{
    height: 26px;
    width: 26px;
    /*background: url("../../../static/image/menu.png") center/cover no-repeat;*/
    position: absolute;
    left: 43px;
    top: 0;
  }
  .advantageCon>ul>li:nth-child(2) .subPicture{
    height: 30px;
    width: 30px;
  }
  .subPicture>img{
    width: 100%;
    height: 100%;
  }
  /***********************************/
  /*ul{
    list-style: none;
  }*/
  .color-box{
    width: 100%;
    height: 81px;
    list-style: none;
  }
  .color-box>li{
    width: 23%;
    height: 100%;
    float: left;
    /*background-color: yellow;*/
    margin-left: 1.6%;
    margin-bottom: 10px;
    border-radius: 4px;
    overflow: hidden;
  }
  .small-box{
    width: 100%;
    height: 51px;
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
  .mask {
    background-color: rgba(0,0,0,0.5);
    /*opacity: 0.4;*/
    position: fixed;
    top: 20%; /*偏移*/
    /*margin-top: -00px;*/
    left:50%;
    margin-left: -700px;
    width: 1400px;
    height: 580px;
    z-index: 1
  }
  .mask>ul{
    width: 86%;
    height: 80%;
    margin: 0 auto;
    margin-top: 60px;
  }
  .mask>ul>li{
    float: left;
    width: 20%;
    height: 100%;
    color: #fff;
    font-weight: bold;
    padding-top: 400px;
    box-sizing: border-box;
  }
  .inlineBlock{
    /*display: inline-block;*/
    float: right;
    /*margin-right: 20%;*/
  }
  .el-input__inner{
    background-color: transparent;
  }
  input.el-range-input{
    background-color: transparent!important;
  }
  el-input__inner .el-range-separator {
    color: #DCDFE6!important ;
  }
</style>
