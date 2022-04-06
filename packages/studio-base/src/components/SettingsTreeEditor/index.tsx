// This Source Code Form is subject to the terms of the Mozilla Public
// License, v2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/

import ClearIcon from "@mui/icons-material/Clear";
import SearchIcon from "@mui/icons-material/Search";
import { AppBar, IconButton, TextField, styled as muiStyled, List, Divider } from "@mui/material";
import { useState } from "react";

import { SettingsTree } from "@foxglove/studio-base/components/SettingsTreeEditor/types";
import Stack from "@foxglove/studio-base/components/Stack";

import { NodeEditor } from "./NodeEditor";

const StyledAppBar = muiStyled(AppBar, { skipSx: true })(({ theme }) => ({
  top: -1,
  zIndex: theme.zIndex.appBar - 1,
  borderBottom: `1px solid ${theme.palette.divider}`,
  padding: theme.spacing(1),
}));

export default function SettingsTreeEditor({
  settings,
  updater,
}: {
  settings: SettingsTree;
  updater: (path: string[], value: unknown) => void;
}): JSX.Element {
  const [filterText, setFilterText] = useState<string>("");

  return (
    <Stack fullHeight>
      {settings.showFilter === true && (
        <StyledAppBar position="sticky" color="default" elevation={0}>
          <TextField
            onChange={(event) => setFilterText(event.target.value)}
            value={filterText}
            variant="filled"
            fullWidth
            placeholder="Filter by layer name"
            InputProps={{
              startAdornment: <SearchIcon fontSize="small" />,
              endAdornment: filterText && (
                <IconButton
                  size="small"
                  title="Clear search"
                  onClick={() => setFilterText("")}
                  edge="end"
                >
                  <ClearIcon fontSize="small" />
                </IconButton>
              ),
            }}
          />
        </StyledAppBar>
      )}
      <List dense disablePadding>
        <Divider />
        <NodeEditor path={[]} settings={settings.settings.tree} updateSettings={updater} />
        <Divider />
      </List>
    </Stack>
  );
}
