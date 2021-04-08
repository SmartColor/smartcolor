<template>
  <div>
<!--    <h1>随机生成</h1>-->
    <div class="contain">
      <ul class="color-box">
        <li>
          <div class="color" :style="{'background-color': color1}"></div>
          <div class="color-tools">
            <h4 class="color-value">{{color1}}</h4>
            <a href="javascript:(void 0)" class="btn btn-lock" ><i class="el-icon-unlock"></i></a>
            <a href="javascript:(void 0)" class="btn btn-operation"  @click="colorOperation()"><i class="el-icon-s-operation"></i></a>
          </div>
        </li>
        <li>
          <div class="color" :style="{'background-color': color2}"></div>
          <div class="color-tools">
            <h4 class="color-value">{{color2}}</h4>
            <a href="javascript:(void 0)" class="btn btn-lock" ><i class="el-icon-unlock"></i></a>
            <a href="javascript:(void 0)" class="btn btn-operation"  @click="colorOperation()"><i class="el-icon-s-operation"></i></a>
          </div>
                    <input type="color" id="color" style="display:none">
        </li>
        <li>
        <div class="color" :style="{'background-color': color3}"></div>
        <div class="color-tools">
          <h4 class="color-value">{{color3}}</h4>
          <a href="javascript:(void 0)" class="btn btn-lock" @click="toggle1()"
             v-bind:class="{ active: isshow1 }"><i class="el-icon-unlock"></i></a>
          <a href="javascript:(void 0)" class="btn btn-operation"  @click="colorOperation()"><i class="el-icon-s-operation"></i></a>
        </div>
        <!--          <input type="color">-->
      </li>
        <li>
        <div class="color" :style="{'background-color': color4}"></div>
        <div class="color-tools">
          <h4 class="color-value">{{color4}}</h4>
          <a href="javascript:(void 0)" class="btn btn-lock" @click="toggle2()"
             v-bind:class="{ active: isshow2 }"><i class="el-icon-unlock"></i></a>
          <a href="javascript:(void 0)" class="btn btn-operation"  @click="colorOperation()"><i class="el-icon-s-operation"></i></a>
        </div>
        <!--          <input type="color">-->
      </li>
        <li>
        <div class="color" :style="{'background-color': color5}"></div>
        <div class="color-tools">
          <h4 class="color-value">{{color5}}</h4>
          <a href="javascript:(void 0)" class="btn btn-lock"><i class="el-icon-unlock"></i></a>
          <a href="javascript:(void 0)" class="btn btn-operation"
             @click="colorOperation()"><i class="el-icon-s-operation"></i></a>
        </div>
        <!--          <input type="color">-->
      </li>
<!--        <input type="color">-->

      </ul>
      <div class="generate-box">
        <div class="generate">
          <a @click="generate"><i class="el-icon-refresh"></i>&nbsp;&nbsp;generate</a>
        </div>
        <div class="description">

        </div>
      </div>
    </div>
    <div class="banner"></div>
  </div>
</template>

<script>
export default {
  name: 'Generate',
  data () {
    return {
      isshow1: false,
      isshow2: false,
      color1: '#b0d0d3',
      color2: '#c08497',
      color3: '#f7af9d',
      color4: '#f7e3af',
      color5: '#f3eec3'
    }
  },
  methods: {
    colorOperation: function () {
      let colorInput = document.getElementById('color')
      colorInput.click()
      var self = this
      colorInput.onchange = function () {
        console.log(this.value)
        self.color4 = this.value
      }
    },
    toggle1: function () {
      this.isshow1 = !this.isshow1
    },
    toggle2: function () {
      this.isshow2 = !this.isshow2
    },
    generate: function () {
      this.color1 = '#ffac81'
      this.color2 = '#ff928b'
      /* this.color3 = '#CED0A1'
      this.color4 = '#FF9770' */
      this.color5 = '#cdeac0'
    }
  }

}
</script>

<style scoped>
  ul{
    list-style: none;
  }
  .contain{
    width: 60%;
    height: 336px;
    /*background-color: yellow;*/
    margin: 0 auto;
  }
  .color-box{
    width: 100%;
    height: 225px;
    margin-top: 50px;
    margin-bottom: 0;
  }
  .color-box>li{
    width: 20%;
    height: 100%;
    float: left;
  }

  .color{
    width: 100%;
    height: 171px;
  }
  .color-tools{
    width: 100%;
    height: 54px;
    text-align: left;
    /*padding: 0 5px;*/
  }
  .color-tools>h4{
    display: inline-block;
    line-height: 54px;
    margin: 0;
    color: #c3c3c3;
  }
  .btn{
    color: #676464;
    font-weight: bold;
    margin-left: 10px;
    padding: 4px 6px;
    border-radius: 100%;
  }
  .active{
    color:#fff;
    background-color: #BEBFB0;
    font-weight: bold;
  }
  .generate-box{
    width: 100%;
    height: calc(100% - 225px);
    display: flex;
    justify-content: left;

  }
  .generate{
    width: 20%;
    height: 100%;
    box-sizing: content-box;
    display: flex;
    justify-content: left;
  }
  .generate>a{
    display: inline-block;
    height: 20px;
    /*width: 85px;*/
    padding: 10px 28px ;
    background-color:#BBBBBB;
    border-radius: 24px;
    text-align: left;
    color: #000;
    border: 0;
    cursor: pointer;
    font-size: 14px;
    font-weight: 400;
    line-height: 1.42857143;
    text-align: center;
    white-space: nowrap;
    vertical-align: middle;
    touch-action: manipulation;
    margin-top: 25px;
    /*margin-left: -115px;*/
  }
  .generate>a:hover{
    color: #333;
    background-color: #e6e6e6;
    border-color: #adadad;
  }
  .generate i{
    font-weight: bolder;
  }
  /*.description{
    width: ;
  }*/

  .banner{
    width: 100%;
    height: calc(100% - 396px);
    background:url("../../../static/imgs/banner1.jpg") center no-repeat;
    height: 262px;
  }
</style>
