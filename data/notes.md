Artem Russian Federation, http://www.insecam.org/en/view/892250/, +10:00
Launceston, Australia, http://45.248.50.164/view/viewer_index.shtml?id=4793, +11:00
Launceston, Australia, http://45.248.50.164:82/view/viewer_index.shtml?id=1231



+12:00

http://www.insecam.org/en/view/760911/
http://www.insecam.org/en/view/760913/
http://www.insecam.org/en/view/762012/
http://www.insecam.org/en/view/802079/
http://www.insecam.org/en/view/813107/
http://www.insecam.org/en/view/852204/


+13:00

http://www.insecam.org/en/view/1007354/
http://www.insecam.org/en/view/897822/
http://www.insecam.org/en/view/898463/
http://www.insecam.org/en/view/906144/
http://www.insecam.org/en/view/908954/
http://www.insecam.org/en/view/974484/
http://www.insecam.org/en/view/427325/
http://www.insecam.org/en/view/451528/
http://www.insecam.org/en/view/634466/
http://www.insecam.org/en/view/811693/
http://www.insecam.org/en/view/817960/
http://www.insecam.org/en/view/846271/
http://www.insecam.org/en/view/328598/

- could be anywhere in the world, but I think they are in the US
http://www.insecam.org/en/view/1007869/
http://www.insecam.org/en/view/881796/
http://www.insecam.org/en/view/882406/
http://www.insecam.org/en/view/882407/
http://www.insecam.org/en/view/882408/
http://www.insecam.org/en/view/884495/
http://www.insecam.org/en/view/869602/
http://www.insecam.org/en/view/870260/
http://www.insecam.org/en/view/877974/
http://www.insecam.org/en/view/879044/
http://www.insecam.org/en/view/881792/
http://www.insecam.org/en/view/881793/
http://www.insecam.org/en/view/798681/
http://www.insecam.org/en/view/809558/
http://www.insecam.org/en/view/817795/
http://www.insecam.org/en/view/851953/
http://www.insecam.org/en/view/864087/
http://www.insecam.org/en/view/869601/
http://www.insecam.org/en/view/635068/
http://www.insecam.org/en/view/635294/
http://www.insecam.org/en/view/724570/
http://www.insecam.org/en/view/724594/
http://www.insecam.org/en/view/742881/
http://www.insecam.org/en/view/746928/


get http://www.insecam.org/en/bytimezone/+01:00/

find all links on page with 'view' in them and save them to the dataframe under column 'url' and with '+01:00' in the 'timezone' column
save one row per link.

then do the same for all timezones.
http://www.insecam.org/en/bytimezone/{timezonecode}/

the element you want to target within each camera page is the id='stream'
the url for the stream is in the src attribute of the element.




```html
<video>
 <source src="http://80.56.142.202:83/mjpg/video.mjpg">
</video>
<a rel="nofollow" href="http://80.56.142.202:83/" target="new"><img id="image0" src="http://80.56.142.202:83/mjpg/video.mjpg" class="img-responsive img-rounded detailimage" alt="" title="Click here to enter the camera located in Netherlands, region Noord-Brabant, Эйндховен"></a>

<script>
var ch = 0;
var imageurls = new Array();
imageurls[0] = new String("http://80.56.142.202:83/mjpg/video.mjpg");
</script>
<script async="" src="https://pagead2.googlesyndication.com/pagead/js/adsbygoogle.js"></script>
<ins class="adsbygoogle" style="display:block; text-align:center;" data-ad-layout="in-article" data-ad-format="fluid" data-ad-client="ca-pub-9642036526375612" data-ad-slot="8439664353"></ins>
<script>
     (adsbygoogle = window.adsbygoogle || []).push({});
</script>
```

So, if you find the `nofollow` a rel and retrieve the href attribute, you will get the url for the stream.
The `title` attribute contains the location of the camera which is also useful to keep in the dataframe.

```html


	<div class="camera-details__row">
		<div class="camera-details__cell">
			Country:
		</div>
		<div class="camera-details__cell">
			<a class="camera-details__link" rel="index" title="Earth webcams in New Zealand" href="/en/bycountry/NZ/">New Zealand</a>
		</div>
	</div>
	<div class="camera-details__row">
		<div class="camera-details__cell">
			Country code:
		</div>
		<div class="camera-details__cell">
			NZ
		</div>
	</div>
	<div class="camera-details__row">
		<div class="camera-details__cell">
			Region:
		</div>
		<div class="camera-details__cell">
			<a class="camera-details__link" rel="index" title="Live camera in  Canterbury" href="/en/byregion/NZ/Canterbury/">Canterbury</a>
		</div>
	</div>
	<div class="camera-details__row">
		<div class="camera-details__cell">
			City:
		</div>
		<div class="camera-details__cell">
			<a class="camera-details__link" rel="index" href="/en/bycity/Christchurch/" title="View online network cameras in Christchurch"> Christchurch</a>
		</div>
	</div>
	<div class="camera-details__row">
		<div class="camera-details__cell">
			Latitude:
		</div>
		<div class="camera-details__cell">
			-43.533330
		</div>
	</div>
	<div class="camera-details__row">
		<div class="camera-details__cell">
			Longitude:
		</div>
		<div class="camera-details__cell">
			172.633330
		</div>
	</div>
	<div class="camera-details__row">
		<div class="camera-details__cell">
			ZIP:
		</div>
		<div class="camera-details__cell">
			8140
		</div>
	</div>
	<div class="camera-details__row">
		<div class="camera-details__cell">
			Timezone:
		</div>
		<div class="camera-details__cell">
			<a class="camera-details__link" rel="index" title="Watch cams in  Canterbury" href="/en/bytimezone/+13:00/">+13:00</a>
		</div>
	</div>
	<div class="camera-details__row">
		<div class="camera-details__cell">
			Manufacturer:
		</div>
		<div class="camera-details__cell">
			<a class="camera-details__link" rel="tag" title="All Hi3516 online cameras directory" href="/en/bytype/Hi3516/">Hi3516</a>
		</div>
	</div>



```