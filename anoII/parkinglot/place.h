#pragma once

struct Place
{
	union
	{
		struct
		{
			int x01, y01;
			int x02, y02;
			int x03, y03;
			int x04, y04;
		};
		int coords[8];
	};
};
