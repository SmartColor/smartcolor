<template>
    <el-header style="height: 85px;">
      <div class="headerCon">
        <div class="title">
          <a href="/website"><img src="../../static/image/menu1.png" alt=""><!--<i class="el-icon-s-help"></i>-->&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;ColorBlend</a>
        </div>

        <el-menu
          router
          :default-active="activeIndex"
          class="el-menu-demo"
          mode="horizontal"
          @select="handleSelect"
          background-color="#111111"
          text-color="#c3c3c3"
          border-bottom="solid 0px transparent"
          active-text-color="#ffd04b">
          <el-menu-item index="/website" style="height:85px;line-height: 85px"><i
            class="el-icon-s-grid"></i>Website
            Colors</el-menu-item>
          <el-menu-item index="/explore" style="height:85px;line-height: 85px"><i
            class="el-icon-s-promotion"></i>
            Explore</el-menu-item>
          <el-menu-item index="/generate" style="height:85px;line-height: 85px"><i class="el-icon-s-open"></i>
          Generate</el-menu-item>
    <!--      <el-menu-item index="/generate1" style="height:85px;line-height: 85px"><i class="el-icon-s-open"></i>
            随机4000</el-menu-item>
          <el-menu-item index="/generate2" style="height:85px;line-height: 85px"><i class="el-icon-s-open"></i>
            fff</el-menu-item>
          <el-menu-item index="/generate3" style="height:85px;line-height: 85px"><i class="el-icon-s-open"></i>
            最新生成</el-menu-item>-->
        </el-menu>

      </div>
      <div class="photo" @click="changeIt">
        <img src="../../static/image/photo.jpg" alt="">
      </div>
      <div class="phptoHover" :style="{display:activeDisplay}">
        <el-menu
          router
          default-active="/huiyang"
          class="el-menu-vertical-demo"
          @open="handleOpen"
          @close="handleClose">
          <el-menu-item index="/huiyang">
            <i class="el-icon-star-on"></i>
            <span slot="title">Huiyang</span>
          </el-menu-item>
          <el-menu-item index="2">
            <i class="el-icon-user-solid"></i>
            <span slot="title">Account</span>
          </el-menu-item>
          <el-menu-item index="/login">
            <i class="el-icon-eleme"></i>
            <span slot="title">Sign out</span>
          </el-menu-item>
        </el-menu>
      </div>
      <div :class="[isActive?'change':'changs']">
        <!--通过点击按钮此处样式会发生改变-->
      </div>
    </el-header>
</template>

<script>
import {mapState} from 'vuex'// 引用action和state映射函数
export default {
  name: 'Header',
  data () {
    return {
      // activeIndex: '/website'
      isActive: false,
      activeDisplay: 'none'
    }
  },
  computed: {
    ...mapState({
      activeIndex: state => state.currentIndex// 获取当前激活菜单
    })
  },
  methods: {
    // 注销用户
    logout: function () {
      this.logoutAction()
      this.$router.push({path: '/login'})
    },
    handleSelect (key, keyPath) {
      console.log(key, keyPath)
    },
    changeIt: function () {
      this.isActive = !this.isActive
      if (this.isActive === true) {
        this.activeDisplay = 'block'
      } else {
        this.activeDisplay = 'none'
      }
    },
    handleOpen (key, keyPath) {
      console.log(key, keyPath)
    },
    handleClose (key, keyPath) {
      console.log(key, keyPath)
    }

  },
  mounted () {

  },
  watch: {

  }
}
</script>

<style scoped>
  .el-menu.el-menu--horizontal {
    border-bottom: solid 0px #e6e6e6;
  }
  .el-header{
    /*font: normal 1em 'Quicksand',sans-serif;*/
    /*font-family:sans-serif;*/

    font-family:Roboto,Helvetica Neue,Helvetica,Arial,sans-serif
  }
  .el-menu-item {
    font-size: 16px;
  }
.headerCon{
  width: 60%;
  float: left;
  /*margin: 0 auto;*/
  margin-left: 19%;
  display: flex;
  justify-content: space-around;
}
  .title>a{
    text-decoration: none;
    color: #c3c3c3;
    font-size: 22px;
    line-height: 85px;
    position: relative;
  }
  .title img{
    width: 30px;
    height: 30px;
    object-fit:cover;
    position: absolute;
    top: -4px;
  }
  .photo{
    float: right;
    width: 45px;
    height: 45px;
    line-height: 85px;
    /*vertical-align: center;*/
    overflow: hidden;
    border-radius:50px;
    margin-top: 20px;
    /*display: inline-block;*/
  }
  .photo>img{
    width: 45px;
    height: 45px;
    object-fit:cover;
    float: left;
    /*line-height: 85px;*/
  }
  .phptoHover{
    width: 140px;
    height: 168px;
    border: 1px solid #cccccc;
    position: absolute;
    right: 0;
    top: 100px;
    overflow: hidden;
    -webkit-border-radius: 5px;
    -moz-border-radius: 5px;
    border-radius: 5px;
    /*display: none;*/
    z-index: 1000;
  }
</style>
<style>

</style>
