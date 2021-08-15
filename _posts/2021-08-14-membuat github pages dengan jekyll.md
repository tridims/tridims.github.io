---
layout: article
title: Creating Github Pages with Jekyll
tags: github-pages, jekyll, first-websites
lang: en
---

## What is Github Pages ?

Github pages is a service from github for hosting a static web pages. This service work by saving the html, javascript, and css directly to github repository and use it to built the website.

Github pages is free, but we can only create one github pages per account. When we create github pages we will get default domain that is username.github.io

## What is Jekyll ?

Jekyll is one of static website generator that was written on ruby. It is pretty popular, and one of the reason is because github support/using it.

Static website generator is a module/library that can create static website from and usually markdown file. It works by converting markdown file to html and apply some theme that we choose.

Jekyll is the kind of static website generator that Github can work with automatically. And by that i mean github can compile or run jekyll in the repository itself to create the website. This is different from other. When we use other static website generator for creating Github Pages, we must run the library and save the html, css, and javascript ourselves then we can put it on the repository.

Jekyll is written on ruby, and we must install ruby and bundler to use this library.

## Step that i took to create my Github Pages

---

### **Install Ruby, bundler, and Jekyll**

The first thing is to install ruby and bundler. This is necessary when you want to create the github pages yourself or you want to test the page in your own machine before commiting changes to github repository. If you not, you can just use follow the step like i did.

I am using ArcoLinux (arch based) so the comman i use to install ruby is:

```shell
sudo pacman -S ruby
gem install bundler
gem install bundler jekyll
```

### **Creating github pages repository**

The next step is to create the repository where the gihub pages will reside. We can do that by creating repository in this format :

> `<user>.github.io`

You must set the visibility of the repository to public if you use github free account, otherwise if it pro account you can put it in the private.

And then the next step is to push the jekyll project to this repository and the repository to be github pages by choosing which branch you want to display.

### **Creating jekyl project**

The next step is to create the jekyll project. We can do this from scratch by first initializing the project using this command.

```zsh
  jekyll new my-awesome-site
  cd my-awesome-site
  bundle exec jekyll serve
  # Now browse to http://localhost:4000
```

You can then choose your theme and start configuring your project. The way to do it you can find it in [here](https://docs.github.com/en/pages/setting-up-a-github-pages-site-with-jekyll/creating-a-github-pages-site-with-jekyll)

That is one way to create the project from scratch, you can learn more from doing that. But if you dont want to, you can just fork or downloading other project/theme/template and modify it.

### **Creating Jekyll project using pre-made theme/template**

For this github pages, i use theme that is called TeXt. You can accest this repository in [here](https://github.com/kitian616/jekyll-TeXt-theme).

You can just download the source code from there, extract it in your project repository. And then customizing it by following the pretty well made documentation.

### **Push the project to the Github Repository and configuring branch in the github pages**

The last step is just to push the project to the repository that we made before, and configuring the github pages settings. To configure that, you can just go to the repository -> Settings -> Pages and then set which branch you want to use as github pages. And Done, you can see your website in the link : `<user>.github.io`
