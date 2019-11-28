---
layout: page
permalink: /poetry/
title: Blog
description: Just the things I do in my free time
---

<ul class="post-list">
{% for poem in site.poetry reversed %}
    <li>
        <h2><a class="poem-title" href="{{ poem.url | prepend: site.baseurl }}">{{ poem.title }}</a></h2>
          <h4 class="post-list">{{ poem.description }}</h4>
      </li>
{% endfor %}
</ul>
