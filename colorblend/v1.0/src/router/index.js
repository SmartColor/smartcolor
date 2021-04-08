import Vue from 'vue'
import Router from 'vue-router'
import Login from '@/components/Account/Login'
import Home from '@/components/Home'
import PageNotFound from '@/components/PageNotFound'

import Website from '@/components/Website/Website'
import Explore from '@/components/Explore/Explore'
import Generate from '@/components/Generate/Generate'
import Huiyang from '@/components/Huiyang/Huiyang'

import Generate1 from '@/components/Generate/Generate1'
import Generate2 from '@/components/Generate/Generate2'
import Generate3 from '@/components/Generate/Generate3'

Vue.use(Router)

export default new Router({
  mode: 'history',
  routes: [
    {
      path: '/',
      redirect: '/home',
      hidden: true
    },
    {
      path: '/login',
      name: 'Login',
      component: Login
    },
    {
      path: '/home',
      name: 'Home',
      redirect: {name: 'Website'},
      component: Home,
      meta: {requireAuth: true}, // 必须要登录才能跳转
      children: [
        /* 违法认定  行人 */
        {
          path: '/website',
          name: 'Website',
          component: Website,
          meta: {isMenuItem: true}
        },
        {
          path: '/explore',
          name: 'Explore',
          component: Explore,
          meta: {isMenuItem: true}
        },
        {
          path: '/generate',
          name: 'Generate',
          component: Generate,
          meta: {isMenuItem: true}
        },
        {
          path: '/huiyang',
          name: 'Huiyang',
          component: Huiyang,
          meta: {isMenuItem: true}
        },
        {
          path: '/generate1',
          name: 'Generate1',
          component: Generate1,
          meta: {isMenuItem: true}
        },
        {
          path: '/generate2',
          name: 'Generate2',
          component: Generate2,
          meta: {isMenuItem: true}
        },
        {
          path: '/generate3',
          name: 'Generate3',
          component: Generate3,
          meta: {isMenuItem: true}
        }
        /** **** 一级菜单选项路由在这里加******/
        // hierarchy management
        /** **********************/
      ]
    },

    {
      path: '*',
      name: 'PageNotFound',
      component: PageNotFound,
      meta: {
        title: '404 页面未找到'
      }
    }
  ]
})
