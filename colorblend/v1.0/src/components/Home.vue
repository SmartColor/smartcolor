<template>
    <el-container direction="vertical">
        <Header/>
        <el-container>
<!--            <Aside/>-->
            <el-main class="main">
                <transition mode="out-in" name="fade">
                    <!--************************ROUTER************************-->
                    <router-view/>
                    <!--******************************************************-->
                </transition>
            </el-main>
        </el-container>
        <Footer/>
    </el-container>

</template>

<script>
import Header from './Header'
import Footer from './Footer'
// import Aside from './Aside'

export default {
  name: 'Home',
  components: {Header, Footer},
  methods: {}
}
</script>

<style scoped>
    .el-footer {
        background-color: #111111;
        color: #c3c3c3;
        text-align: center;
        line-height: 60px;
    }
   .el-header{
     background: #111111;
     color: #c3c3c3;
     width: 100%;
     text-align: center;
     height:85px;
     text-align: left;
     font-size: 14px;
     /*display: flex;*/
     /*justify-content: center;*/
     /*line-height: 70px;*/
     padding: 0 60px;

   }
    .el-main {
        /*background-color: #E9EEF3;*/
        min-height: 647px;
        /*background-color: #FDFDFD;*/
        background-color: #0B0C0E;
        color: #333;
        text-align: center;
        padding: 0px;
        /*padding-top: 0px;*/
    }

    body > .el-container {
        /*margin-bottom: 40px;*/
    }

</style>
