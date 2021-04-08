<template>
  <div>
    <div class="website">
      <div class="con">
        <div id="profile_info">
          <div class="avatar avatar--m _m-b-4" id="header_profile-avatar"><div class="image" style="background-image:
           url('../../../static/image/photo.jpg')"></div></div>
          <h1>huiyang</h1>
        </div>
      </div>

    <div id="profile_palettes">
      <ul>
        <li style="backgroundColor:rgb(82, 232, 246)">#52E8F6</li>
        <li style="backgroundColor:rgb(13, 154, 255)">#0D9AFF</li>
        <li style="backgroundColor:rgb(4, 138, 129)">#048A81</li>
        <li style="backgroundColor:rgb(222, 156, 154)">#DE9C9A</li>
        <li style="backgroundColor:rgb(133, 108, 107)">#856C6B</li>
      </ul>
      <div class="tip"></div>
    </div>
    </div>
    <!--<div class="advantage">
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
&lt;!&ndash;当我们需想要更多的配色时，根据我们生成的大量配色数据，通过锁住其中一个或者两个、三个、四个颜色来生成其余的颜色，为用户提供多种配色选择&ndash;&gt;
          </ul>
       </div>
    </div>-->
  </div>
</template>

<script>
export default {
  name: 'Website',
  data () {
    return {

    }
  },
  methods: {

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
    background-color: #F4F5EF;
    padding: 60px 0px 60px 0;
    height: 528px;
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

  /*个人收藏样式*/
  #profile_info {
    text-align: center;
    padding-bottom: 60px;
  }

  #header_profile-avatar{
    position: relative;
    border-radius: 50%;
    vertical-align: middle;
    height: 140px;
    width: 140px;
    margin: 0 auto;
  }
  #header_profile-avatar>.image {
    background-size: cover;
    background-position: center;
    /*background-color: #fff;*/
    background-image: url('../../../static/image/photo.jpg');
    position: absolute;
    left: 0;
    top: 0;
    width: 100%;
    height: 100%;
    border-radius: 50%;
    box-shadow: inset rgba(0,0,0,.11) 0 0 0 1px;
  }
  #profile h1 {
    margin-bottom: 0;
  }
  h1 {
    font-size: 50px;
    margin-bottom: 30px;
    font-weight: 600;
  }
  h1, h2, h3, h4 {
    color: #000;
    line-height: 1.1em;
    letter-spacing: -.035em;
  }
  #profile_palettes{
    width: 60%;
    border-radius: 14px;
    overflow: hidden;
    margin: 0 auto;
    border: 1px solid #ccc;
  }
  #profile_palettes>ul{
    width: 100%;
    height: 134px;
    margin: 0px;
  }
  #profile_palettes>.tip{
    width: 100%;
    height: 30px;
    background-color:#fff;
    /*border-radius: 14px 14px 0 0;*/
    /*overflow: hidden;*/
  }
  #profile_palettes>ul>li{
    float: left;
    width: 20%;
    height: 100%;
    color: #fff;
    font-weight: bold;
    /*padding-top: 400px;*/
    box-sizing: border-box;
    line-height: 134px;
  }
</style>
